# 演示与实验记录（可复现）

> 所有输出都是本机实跑结果（RTX 5060 Laptop 8GB + Ultra 5 135H + 本地 Ollama）。
> 命令在每节末尾，可原样复现。

## 1. 离线演示（不碰微信、不联网、不发送）

```powershell
python -m wagent.main --mock --fake-llm
```

```
[memory] 向量索引：{'episode': 0, 'style': 2, 'fact': 1}
[*] backend=mock model=fake mode=dry_run interval=0.0s
[18:18:24] #1   无新消息
[18:18:24] #2   [预览·未发送] 在的 八点老地方见
[18:18:24] #3   无新消息          ← 自己的回复出现在屏幕上，不重复回
[18:18:24] #4   [预览·未发送] 在的 八点老地方见
[18:18:24] #5   无新消息
```

## 2. 真机 dry-run + trace（默认不发送）

```powershell
python -m wagent.main --fake-llm --ticks 3 --interval 0 --trace
```

```
[*] backend=wechat model=fake mode=dry_run interval=1.0s
[18:19:09] #1   无新消息
[18:19:09] #2   无新消息
[18:19:09] #3   无新消息
[trace] 汇总：{'events': 6, 'counts': {'perceive': 3, 'tick': 3},
              'total_ms': {'perceive': 12038.1, 'tick': 12039.8}, 'errors': 0}
```

trace 明细（冷启动 vs 帧缓存命中）：

```json
{"seq": 1, "kind": "perceive", "ms": 11957.4, "messages": 12}
{"seq": 3, "kind": "perceive", "ms": 41.3,  "messages": 12}
{"seq": 5, "kind": "perceive", "ms": 39.4,  "messages": 12}
```

## 3. 层级记忆 + 向量召回（本地 bge-m3）

```powershell
python -m wagent.import_chat csv --file data/raw/demo_chat.csv --person 张伟
python scripts/consolidate.py --db data/demo_agent.db --person 张伟 --index --no-summary
```

```
[index] 新增向量：{'episode': 1, 'style': 0, 'fact': 6}，累计 7 条
```

语义召回验证（3 个月前的旧事件）：

```
--- query: 他工作怎么样了 实习有消息吗
- 08-01 他说想换工作，投了字节的实习，怕过不了     ← 语义命中排第一
- 11-02 对方下个月搬去滨江，周六帮搬家…
--- query: 你搬家的事定下来了吗
- 11-02 对方下个月搬去滨江，周六帮搬家…            ← 近期事件排第一
```

## 4. 3B LoRA A/B 评测（base vs adapter）

```powershell
python scripts/train_lora.py --synth --data data/train_synth --out data/lora/qwen25-3b-style
python scripts/eval_ab.py --adapter data/lora/qwen25-3b-style --data data/train_synth/val.jsonl
python scripts/judge_replies.py        # 生成结束后单独跑，避开显存争用
```

| 指标 | base | LoRA |
|---|---|---|
| 综合分 | 0.455 | **0.510** |
| 风格相似度 | 0.014 | 0.020 |
| AI 味 | 0.21 | **0.00** |
| 长度自然度 | 1.00 | 1.00 |
| LLM 盲评（10 例） | 4 胜 | **5 胜**（1 平，胜率 55.6%） |

评委典型样例：

```
base  : 当然可以，把具体的代码和问题描述给我，我来帮你搞定。
LoRA  : 你先说功能需求
base  : 哎呀，那可真不错，新工作咋样？
LoRA  : 挺好的
```

> ⚠️ 训练数据是**合成数据**（228 条），指标只证明链路可用；真实效果需真实聊天记录重训。

## 5. 消融实验：有 / 无层级记忆

```powershell
python scripts/ablation.py --limit 12
```

```
[ablation] 向量索引：{'episode': 2, 'style': 15, 'fact': 3}
[ablation] 12 条样本，记忆命中率=1.0，模型=qwen2.5:7b-instruct-q4_K_M
[ablation] 无记忆：{'n': 12, 'score': 0.587, 'style': 0.189, 'ai_flavor': 0.025, 'length': 0.994}
[ablation] 有记忆：{'n': 12, 'score': 0.625, 'style': 0.251, 'ai_flavor': 0.0,   'length': 1.0}
[ablation] 盲评：无记忆 5 / 有记忆 3 / 平 4（有记忆胜率 0.375）
```

**怎么读这个结果（诚实结论）**：

- 规则指标上记忆组更好：风格相似 0.189→0.251、AI 味 0.025→0.0、综合分 0.587→0.625；
- 但 12 条**合成样本**上的 LLM 盲评没有显著差异（样本太小 + 合成 prompt 与预置记忆内容不匹配），
  说明「记忆有没有用」必须用真实聊天数据验证，不能拿合成数据自嗨；
- 这本身是评测体系的价值：**它没有替项目吹结论，而是暴露了问题**。

## 6. 复现清单

```powershell
pytest -q                              # 200+ 单测
ruff check src tests scripts && mypy   # 静态检查
python scripts/probe_window.py         # 现场诊断：窗口/截图/OCR/输入框/配色
```

产物索引：

| 文件 | 内容 |
|---|---|
| `data/eval/ab_report.md` | LoRA A/B 规则指标 |
| `data/eval/judge_report.md` | LLM 盲评逐条结果 |
| `data/eval/ablation_report.md` | 记忆消融实验 |
| `data/trace/*.jsonl` | 每轮耗时/动作 trace |
| `data/lora/qwen25-3b-style/` | LoRA adapter（本地，不入库） |
