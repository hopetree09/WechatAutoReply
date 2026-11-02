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
[18:26:03] #1   无新消息
[18:26:03] #2   [预览·未发送] 在的 八点老地方见
[18:26:03] #3   无新消息              ← 自己的回复出现在屏幕上，不重复回
[18:26:03] #4   [跳过] 无需回复（语气词/表情）   ← 意图路由：「嗯」不回
[18:26:03] #5   [预览·未发送] 在的 八点老地方见
[18:26:03] #6   无新消息
```

护栏生效时的输出（回复含手机号会自动脱敏，含"转你 200 块"直接拦截）。

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

### 4.1 合成数据（链路验证）

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

### 4.2 真实数据（最终结果）

```powershell
# 1) 导入 Moect 导出的 JSONL（28,869 条真实文本消息，3 年）
python -m wagent.import_chat inspect --file data\raw\<导出目录>\xxx.jsonl --person <联系人>
python -m wagent.import_chat json    --file data\raw\<导出目录>\xxx.jsonl --person <联系人>

# 2) 记忆巩固 + 向量索引（77 条事实 / 4 条事件 / 300 风格样本 / 332 向量）
python scripts/consolidate.py --person <联系人> --limit 1200 --index

# 3) 真实训练集（10,759 对 Q/A，切 9,684 train / 1,075 val）
python scripts/export_training_data.py --person <联系人> --out data/train_real

# 4) 800 条真实样本训练 3B QLoRA（10.5 分钟）
python scripts/train_lora.py --data data/train_real --out data/lora/qwen25-3b-real --limit-samples 800

# 5) A/B 生成 + 盲评
python scripts/eval_ab.py --adapter data/lora/qwen25-3b-real --data data/train_real/val.jsonl --limit 15 --out data/eval/real
python scripts/judge_replies.py --report data/eval/real/ab_report.json --limit 15
```

| 指标 | base | 真实 LoRA |
|---|---|---|
| 综合分 | 0.468 | **0.505** |
| AI 味 | 0.22 | **0.00** |
| 风格相似度（对短回复不敏感） | 0.056 | 0.021 |
| LLM 盲评（15 例） | 1 胜 | **10 胜**（4 平，**胜率 90.9%**） |

评委样例（`data/eval/real/judge_report.md` 有完整 15 条）：

```
base : 好的，我知道了。                                 → 负
LoRA : 嗯嗯                                             → 胜
base : 那我直接去考场了，考试结束后咱们再复习吧。          → 负
LoRA : [强]                                             → 胜
base : 哈哈，都是因为你！下午见，得赶紧去准备一下，不然都来不及了。 → 平（位置偏差抵消）
```

> 真实训练数据：2.9 万条消息（其中我的回复 1.15 万条），平均 8.3 字、中位数 5 字、
> 句号率 0.1%——所以 LoRA 学到的核心就是「极短、口语、不加句号」，盲评也印证了这一点。
> 训练只用了 800 条子集（受限单轮时间），全量 9,684 对可继续加训。

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

**真实数据复跑**（`python scripts/ablation.py --db data/agent.db --person <联系人> --person` 用真实记忆库，
库里有 77 条带证据事实、4 条事件、300 条风格样本）：

```
[ablation] 真实记忆库：data\agent.db，person=<联系人>，风格样本=212
[ablation] 无记忆：{'n': 12, 'score': 0.529, 'style': 0.058, 'ai_flavor': 0.0, 'length': 1.0}
[ablation] 有记忆：{'n': 12, 'score': 0.533, 'style': 0.069, 'ai_flavor': 0.0, 'length': 0.993}
[ablation] 盲评：无记忆 5 / 有记忆 2 / 平 5
```

结论：**短句闲聊场景里记忆注入收益不明显**；记忆真正的价值在「回忆事实/约定」，
应该用回忆类问题单独评测（已列入后续工作）。

## 6. 复现清单

```powershell
pytest -q                              # 全量单测
ruff check src tests scripts && mypy   # 静态检查
python scripts/probe_window.py         # 现场诊断：窗口/截图/OCR/输入框/配色
python -m wagent.import_chat inspect --file <导出文件>   # 探测格式并给出导入命令
```

产物索引（真实数据相关产物含隐私，已在 `.gitignore` 中排除）：

| 文件 | 内容 |
|---|---|
| `data/eval/real/ab_report.md` | 真实 LoRA A/B 规则指标（本地） |
| `data/eval/real/judge_report.md` | 真实 A/B LLM 盲评逐条结果（本地） |
| `data/eval/real2/ablation_report.md` | 真实库记忆消融（本地） |
| `data/trace/*.jsonl` | 每轮耗时/动作 trace（本地） |
| `data/lora/qwen25-3b-real/` | 真实数据 LoRA adapter（本地，不入库） |
