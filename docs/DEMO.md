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
[ablation] 无记忆：{'n': 12, 'score': 0.558, 'style': 0.117, 'ai_flavor': 0.0, 'length': 1.0}
[ablation] 有记忆：{'n': 12, 'score': 0.588, 'style': 0.176, 'ai_flavor': 0.0, 'length': 1.0}
[ablation] 盲评：无记忆 2 / 有记忆 4 / 平 6（有记忆胜率 0.667）
```

结论：修复 `__self__` 事实检索 bug 后，记忆组从"负收益"变成"小幅正收益"（盲评 0.286 → 0.667），
但短句闲聊里平局仍多。记忆真正的价值在「回忆事实/约定」，已由 `scripts/recall_eval.py` 单独验证：
**recall@1 0.61→0.96、recall@3 0.64→1.00、MRR 0.62→0.98**。

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

## 7. 不碰微信的三组自测（画像 / 记忆 / 运行时行为）

```powershell
python scripts/demo_compare.py        # 手写上下文 × 3 组 prompt + MockAdapter 跑完整运行时
```

- 三组：A 裸模型、B +自动画像、C +画像+层级记忆；
- 运行时行为：语气词 → 意图路由跳过；借钱 → 敏感话题跳过；
  「你电脑什么配置」→ 记忆命中后回复「5060的，准备换5090的」；普通约饭 → dry-run 预览不发送。

实测观察（7B，上下文为脚本手写，不含真实聊天）：

**第一轮（记忆无差别注入 + 画像未约束口头禅）**

| 组 | 平均长度 | 观察 |
|---|---|---|
| A 裸模型 | 16.6 字 | 礼貌、爱加「～」、有 AI 味 |
| B +画像 | 18.1 字 | 明显变口语、开始用 `[失望]/[得意]`，但口头禅（「你玩吧」）会不分场合乱用 |
| C +画像+记忆 | 16.9 字 | 能答出「5060 显卡」「周日下午回来」等记忆事实，但闲聊会塞无关事实、偶尔答非所问 |

**第二轮（口头禅改「偶尔使用」+ 记忆按意图注入）**

改动：① system prompt 增加「画像口头禅/表情只是偶尔用，不能每句都带、不能答非所问地硬套」；
② 新增 `needs_memory()`：提问 / 约时间 / 含回忆关键词才注入记忆，语气词/敏感话题/普通闲聊不注入；
③ `build_profile.py` 增加质量门槛（缺小节/含隐私 → 退回规则版画像）与隐私过滤。

| 组 | 平均长度 | 观察 |
|---|---|---|
| A 裸模型 | 26.8 字 | 仍然啰嗦、爱加「～」 |
| B +画像 | **14.4 字** | 口头禅乱用消失，回复明显更短更口语 |
| C +画像+记忆 | 13.9 字 | 记忆注入从 8/8 降到 **5/8**；技术求助/情绪/语气词不再塞记忆，问答类仍能答出「5060」「周日下午回来」 |

> 剩余问题：模型仍会在不合适的时候冒出口头禅（约饭结尾「我草」），情绪安慰场景偶尔答非所问；
> 前者可在上线前做后处理（句尾口头禅过滤），后者是 7B 能力问题，最终建议换 3B LoRA 或 best-of-2。

> 顺带修了一个真 bug：`clean_reply` 的引号清洗集合里包含了 `[]`，
> 会把 `[强]` 洗成 `强`、`[衰]没呢` 洗成 `衰]没呢`（实测输出里真的出现了）。
> 已改为只清洗中英文引号，并加回归测试。

## 8. 3B LoRA 直接吃「画像+记忆」的实测：负结果（重要）

三件套（画像 + 记忆 + LoRA）的设想很自然，但**实测下来 3B LoRA 撑不住**，记录如下。

```powershell
# ① 原始 adapter（短 prompt 训练）直接喂长 prompt（画像+记忆）
python scripts/demo_compare.py --provider lora --adapter data/lora/qwen25-3b-real

# ② 按推理 prompt 重做训练集并重训（system 里带画像+记忆）
python scripts/export_matched_training_data.py --person <联系人> --limit 2000
python scripts/train_lora.py --data data/train_real_matched --out data/lora/qwen25-3b-real-matched --limit-samples 800
python scripts/demo_compare.py --provider lora --adapter data/lora/qwen25-3b-real-matched

# ③ 原始 adapter 用回训练时的短 prompt（验证「纯风格引擎」）
python scripts/demo_compare.py --provider lora --adapter data/lora/qwen25-3b-real --system-mode short --no-runtime
```

| 配置 | 平均长度 | AI 味 | 现象 |
|---|---|---|---|
| 7B + 画像 + 记忆闸门（上一轮） | 13.9 字 | 0.00 | 内容/口吻都还行，能答「5060」「周日下午回来」 |
| 3B 原始 LoRA + 长 prompt | 4~9 字 | 0.00 | 口头禅复读（「我草」）、玩笑直接复述对方原话、答非所问 |
| 3B matched LoRA + 长 prompt | 27~42 字 | **0.19~0.43** | 更糟：开始**复读 system 里的记忆/画像**（「导师是武大女篮队员」重复、编导师对话），长回复退化 |
| 3B 原始 LoRA + 短 prompt | 8.0 字 | 0.00 | 短句口语风格保持，但没有画像/记忆能力，内容常答错（显卡胡编、情绪答非所问） |

原因判断：
1. **train/inference prompt 分布不一致**：训练时 system 只有一句话，推理时是「画像 + 77 条事实 + few-shot」的长 prompt；
2. **3B 容量不够**：长 system 里的事实密度太高，模型学会"抄 prompt"而不是"参考 prompt"；
3. 按推理 prompt 重训（matched，loss 0.32）反而**过拟合到复述 system 内容**——loss 很低不代表回复好，这是个小模型陷阱。

结论与建议（已写进 RESUME 的"局限"）：
- 3B LoRA 的正确定位是**风格引擎**，不要指望它同时做记忆推理；
- 三件套的可行组合是**两段式**：7B/云端负责"理解 + 记忆 + 生成内容" → 3B LoRA 负责"改写成我的口吻"；
- 或上线直接用 7B + 画像 + 记忆闸门（当前已经可用），3B LoRA 留作低成本风格改写器；
- 想单模型端到端，需要用 7B 在长 prompt 下批量生成数据来蒸馏 3B（工程量大，未做）。

> 这次是**负结果**，但它说明评测体系在起作用：如果没有 `demo_compare` + 规则指标 + 盲评，很容易把"loss 0.32"当成成功。

## 9. 两段式（7B 草稿 → 3B LoRA 改写）实测：也是负结果

思路：7B 负责内容/记忆，3B LoRA 负责口吻改写。实现见 `src/wagent/agent/two_stage.py`。

`powershell
python scripts/demo_two_stage.py --stage gen      # 7B 出草稿 → 释放 7B → 3B 改写（同 8 场景）
python scripts/demo_two_stage.py --stage rewrite  # 再试 few-shot 改写 prompt
python scripts/demo_two_stage.py --stage judge    # 3B 进程退出后，7B 盲评（草稿 vs 改写）
```

| 组 | 平均长度 | 规则分 | 盲评（vs 7B 草稿） |
|---|---|---|---|
| 7B 草稿（含画像+记忆） | 11.5 字 | 0.536 | — |
| 3B LoRA 改写（原 prompt） | 5.0 字 | 0.559 | 草稿 6 / 改写 1 / 平 1 |
| 3B LoRA 改写（+few-shot 示例） | 4.4 字 | 0.500 | 草稿 6 / 改写 1 / 平 1 |

典型丢内容：

```
草稿: 行 周日下午我回来了 周边有啥好吃的推荐不
改写: 有的                                      ← 时间/推荐信息全丢
草稿: 5060的～                                  ← 记忆命中
改写: 现在是128GB的                              ← 改出错误内容
草稿: 没有吧～                                   ← 对你是不是又胖了的正确反应
改写: 我胖了
```

原因：3B LoRA 从没在「改写/转换」任务上训练过，它只会像本人那样直接回一句，
因此把草稿当背景噪声。要让它真能改写，必须**专门训练 rewriter**：
用 7B 对真实样本生成草稿 → 构造（草稿+上下文 → 我的真实回复）训练对 → 训 3B。
这是可行的（2000 条草稿约 50 分钟生成 + 800 条约 11 分钟训练），但今天没做。

> 注意：规则分又一次误导——改写组 0.559 > 草稿 0.536，但盲评 6:1 草稿胜。
> 短回复在风格/长度指标上天然占便宜，**别用单一规则分做上线决策**。

> 最终决策：**在线用 7B + 画像 + 记忆闸门**；3B LoRA 只做离线风格改写（输出过 7B 质检）。
> 两段式代码保留在 src/wagent/agent/two_stage.py，作为后续 rewriter 训练的起点。

## 11. 专门训练 rewriter 的实测：风格能学，内容保真学不会（负结果）

按第 9 节的建议，用 7B 对真实样本生成草稿，构造（草稿 → 我的真实回复）训练对，训了两版 rewriter：

| 版本 | 训练配置 | 训练 loss | 8 场景盲评（vs 7B 草稿） | held-out 30 条 |
|---|---|---|---|---|
| v1（上下文+示例 prompt） | 500 对 / 1 epoch / 10.8 min | 1.16 | 草稿 4 / 改写 2 / 平 2（25%） | 保真度 0.00、风格 0.049 |
| v2（极简 prompt，草稿紧贴指令） | 500 对 / 2 epochs / 20 min | 0.86 | — | **保真度 0.01**、风格 0.113、7B 盲评 **46.7% 胜**（草稿 7 / 改写 14 / 平 9） |

关键发现：

1. **3B 学不会「改写」这个条件变换任务**：连训练集自身的保真度都是 0——它把草稿当噪声，重新生成一句"我的口吻"；
2. **风格盲评会骗人**：v2 在 held-out 上盲评胜率 46.7%（评委觉得"更像本人"），但草稿里的时间/数字/结论全丢了
   （"5060的" → "1062.5GB"、"行 周日下午我回来了" → "有的"）——**只看风格的评委发现不了内容错误**；
3. 因此给 `TwoStageGenerator` 加了 **fidelity 兜底**：改写结果与草稿的 2-gram 重叠 < 0.15 且草稿 ≥6 字时，直接退回草稿，
   把"风格洗白"变成"保守改写"（宁可不像、也不能说错），并加了单测。

结论：**500~1000 对数据、8GB 卡上 3B 学不会 rewriter**。要真做，需要
（a）几千对数据 + 更长训练，（b）编辑式目标（保留实体/数字的约束解码或 DPO），（c）或直接用 7B 做 rewriter。
当前保持"在线 7B + 记忆闸门"，3B LoRA 做离线风格工具；rewriter 代码与数据管线已保留，随时可续。
