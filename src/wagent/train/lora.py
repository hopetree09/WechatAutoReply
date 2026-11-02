"""3B QLoRA 风格微调（Qwen2.5-3B-Instruct + PEFT）。

为什么要微调：RAG/画像能解决「记得住」，但「说话像不像我」在短句、标点、口头禅上
更吃模型本身。所以：**知识走记忆库，口吻走 LoRA**。

工程约束（RTX 5060 Laptop / 8GB 显存）：
- 4-bit QLoRA（bitsandbytes）→ 3B 权重约 2GB，激活可控；
- 序列长度 512、batch 1 + 梯度累积 8；
- 只保存 adapter（~50MB），不合并全量权重；
- 训练/推理都用 ``transformers + peft``，不依赖 ollama 的 GGUF。

所有 torch/transformers 导入都放在函数内部：纯逻辑单测不需要装深度学习栈。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass(frozen=True, slots=True)
class LoraConfig:
    base_model: str
    train_file: Path
    output_dir: Path
    val_file: Path | None = None
    epochs: float = 1.0
    lr: float = 2e-4
    batch_size: int = 1
    grad_accum: int = 8
    max_seq_len: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True
    seed: int = 42
    logging_steps: int = 5
    max_samples: int | None = None  # 冒烟测试：只取前 N 条

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ("train_file", "output_dir", "val_file"):
            if data.get(key) is not None:
                data[key] = str(data[key])
        return data


def read_chatml(path: Path) -> list[dict]:
    """读取 ChatML JSONL 并做最小校验。"""
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        messages = item.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"非法样本（缺少 messages）：{line[:80]}")
        if not any(m.get("role") == "assistant" for m in messages):
            raise ValueError(f"非法样本（缺少 assistant 回复）：{line[:80]}")
        rows.append(item)
    if not rows:
        raise ValueError(f"数据集为空：{path}")
    return rows


def format_chatml_fallback(messages: list[dict]) -> str:
    """没有 tokenizer 时给测试用的简化模板。"""
    parts = []
    for message in messages:
        parts.append(f"<|{message.get('role', 'user')}|>\n{message.get('content', '')}\n")
    return "".join(parts)


def texts_from_rows(rows: list[dict], tokenizer=None) -> list[str]:
    """把 ChatML 样本转成训练文本（优先用 tokenizer 的 chat template）。"""
    texts: list[str] = []
    for row in rows:
        messages = row["messages"]
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            text = format_chatml_fallback(messages)
        texts.append(text)
    return texts


# ---------------------------------------------------------------- 训练


def train_lora(config: LoraConfig) -> dict:
    """跑一次 QLoRA 训练，保存 adapter 并返回训练报告。"""
    import torch
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    started = time.time()
    train_rows = read_chatml(config.train_file)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    use_4bit = config.load_in_4bit and torch.cuda.is_available()
    if use_4bit:
        try:
            import bitsandbytes  # noqa: F401 - 触发可用性检查

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        except Exception:
            use_4bit = False
            quantization_config = None

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model: Any = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map={"": 0} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    peft_config = PeftLoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=DEFAULT_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    texts = texts_from_rows(train_rows, tokenizer)

    def _tokenize(batch: dict) -> dict:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.max_seq_len,
            padding=False,
        )
        return encoded

    dataset = Dataset.from_dict({"text": texts}).map(_tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    arguments = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr,
        logging_steps=config.logging_steps,
        save_strategy="no",
        report_to=[],
        seed=config.seed,
        bf16=bf16_supported,
        fp16=torch.cuda.is_available() and not bf16_supported,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )

    trainer = Trainer(model=model, args=arguments, train_dataset=dataset, data_collator=collator)
    train_result = trainer.train()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    report = {
        "config": config.to_dict(),
        "samples": len(train_rows),
        "train_loss": round(float(train_result.training_loss), 4),
        "elapsed_s": round(time.time() - started, 1),
        "use_4bit": use_4bit,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "adapter_dir": str(config.output_dir),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    (config.output_dir / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


# ---------------------------------------------------------------- 推理


def load_for_inference(base_model: str, adapter_dir: str | None = None, *, load_in_4bit: bool = True):
    """加载基座（可选挂 adapter）用于生成；返回 (model, tokenizer)。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model: Any = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={"": 0} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def generate_reply(model, tokenizer, user_content: str, *, max_new_tokens: int = 48) -> str:
    """给定 user 内容（含上下文），生成一句回复。"""
    import torch

    messages = [
        {
            "role": "system",
            "content": "你是「我」本人，正在微信聊天，用我平时的口吻简短回复。",
        },
        {"role": "user", "content": user_content},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
