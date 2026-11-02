# 真实聊天记录采集与导入指南（微信 4.x）

> 目标：把**你自己的**微信聊天记录导出成文件 → 导入本项目的记忆库 / LoRA 训练集。
> 本仓库**不含任何解密代码**：数据怎么来由你决定并承担相应风险；导入、清洗、巩固、训练这些是我们做的事。

---

## 0. 先看清现状（为什么这件事麻烦）

| 事实 | 说明 |
|---|---|
| 微信 4.x 没有官方「导出 CSV」功能 | 只有"聊天记录迁移与备份"，备份出来仍是加密文件 |
| 本地数据库是加密的 | 本机实测：`D:\wechat\liaotianjilu\xwechat_files\wxid_*/db_storage` 下 26 个 `.db` 共 0.60GB，文件头 `bb 0f 05 33…` 不是 SQLite 明文 |
| 所以导出工具都要做同一件事 | 从微信进程内存里取密钥 → 解密本地 SQLCipher 数据库 → 读消息表 |
| 这类工具正处于灰色地带 | 2025-10 腾讯集中发律师函：PyWxDump、chatlog 等知名项目已删库；现在能用的多是新项目/fork |

**结论**：要么用第三方工具导出（完整、精确，但有法务/供应链风险），要么用本项目自带的滚动 OCR 采史（零第三方代码，慢、丢媒体）。

---

## 1. 方案对比

| 方案 | 完整历史 | 精确时间戳 | 图片等媒体 | 风险 | 适合 |
|---|---|---|---|---|---|
| **Moect 导出 JSONL/CSV** | ✅ | ✅（秒级） | ✅（JSONL 带资源路径） | 第三方工具：新项目、需解密数据库 | 想认真训一版 LoRA |
| Moect / 类似工具导出 CSV | ✅ | ✅ | 部分 | 同上，且 CSV 可能没有"是否我发送"列 | 能拿到 JSONL 时优先 JSONL |
| **本项目滚动 OCR 采史** | 看滚多少屏 | 屏幕时间戳（分钟级） | ❌ | **零第三方代码** | 不想碰第三方工具、几百条够用 |
| 微信官方"迁移与备份" | ✅ | ✅ | ✅ | 低 | 只备份，不能直接给本项目用 |

---

## 2. 方案 A：Moect 导出（推荐用于完整历史）

项目主页：<https://github.com/mofan6/Moect>（GPL-3.0，Python + React，v1.2.0，2026-11-02）

### 2.1 下载与校验

```powershell
# 建议放 D:\tools\moect
mkdir D:\tools\moect -Force
curl.exe -L --retry 5 --retry-all-errors -o D:\tools\moect\Moect-1.2.0-Windows-x64.zip `
  https://github.com/mofan6/Moect/releases/latest/download/Moect-1.2.0-Windows-x64.zip

# 必须核对 SHA256（官方 SHA256SUMS.txt 里的值）
Get-FileHash D:\tools\moect\Moect-1.2.0-Windows-x64.zip -Algorithm SHA256
# 期望：a56005ee18a3cfb0ea4adf33341e22ac6f5d33c72a400508772a466de1377748
```

> ⚠️ 如果哈希对不上，**不要运行**，重新下载（网络中断会导致文件不完整）。
> 也建议先开 Windows Defender / 杀软扫描，或在虚拟机里跑。

### 2.2 导出

1. 打开并**登录电脑微信**（工具需要连接微信进程）；
2. 解压/安装 Moect 并启动，点"**连接微信**"；
3. 选择要导出的**好友会话**（先只导一个联系人，别一次全量）；
4. 导出格式选 **JSONL**（首选）或 **JSON**；CSV 也可以但字段更少；
5. 时间范围选"全部"，导出目录默认 `文档\Moect\exports\...`；
6. 导出完成后，把文件复制到本项目的 `data/raw/`（该目录已被 `.gitignore` 排除）。

### 2.3 导入本项目

```powershell
cd D:\wechaautoreply
conda activate wagent

# ① 先看文件长什么样（格式/列名/条数/建议命令）
python -m wagent.import_chat inspect --file data\raw\张伟.jsonl --person 张伟

# ② 导入（JSONL/JSON 字段最全，说话人判定确定）
python -m wagent.import_chat json --file data\raw\张伟.jsonl --person 张伟

# ③ 抽事实 + 建向量索引
python scripts/consolidate.py --person 张伟 --index

# ④ 导出 LoRA 训练集
python scripts/export_training_data.py --output data/train
```

导入完成后会打印：总行数、导入条数、跳过媒体/空行/时间异常的数量，以及库统计（messages/facts/episodes/...）。

### 2.4 CSV 的注意事项

Moect 有两个 CSV 导出路径，表头不同：

- **带「是否我发送」列**（是/否）：直接可用。
- **不带「是否我发送」列**（表头 `时间,会话,发送者,类型,内容,消息ID`）：
  必须用 `--me "你的微信昵称"` 告诉程序哪个发送者是你，否则**所有消息都会被当成对方说的**。
  导入器检测到这种情况会显式告警：

```powershell
python -m wagent.import_chat csv --file data\raw\张伟.csv --person 张伟 --me "我的微信昵称"
```

---

## 3. 方案 B：本项目滚动 OCR 采史（零第三方代码）

```powershell
# 先把目标会话切出来、滚到最新消息
python -m wagent.import_chat scroll --person 张伟 --max-frames 5 --dry-run   # 试采 5 屏
python -m wagent.import_chat scroll --person 张伟 --max-frames 40           # 正式采
```

- 每屏 OCR 约 8 秒，一屏 8~15 条 → 40 屏约 300~600 条、5 分钟左右；
- 只能拿到屏幕上的文字：图片/表情/语音会丢；时间戳精度到屏幕上显示的那一级（分钟/天）；
- 采史期间会移动鼠标滚轮，别同时操作微信；把聊天窗口拉大能一次采更多。

---

## 4. 隐私与安全清单

- [ ] 只处理**你自己的**聊天记录，不导出他人隐私用于其他用途；
- [ ] 导出文件只放本机 `data/raw/`（已在 `.gitignore` 里），**不要**上传网盘/GitHub/发给别人；
- [ ] 第三方工具只在**本机/离线**运行；跑完就关；先备份 `xwechat_files` 再操作；
- [ ] 生产环境别把 `data/raw/`、`data/agent.db`、`data/lora/` 提交进仓库（默认已忽略）；
- [ ] 第三方工具的解密行为可能违反微信用户协议，账号/法律风险自行评估。

---

## 5. 常见问题

| 现象 | 处理 |
|---|---|
| `inspect` 提示"缺少时间列/内容列" | 导出工具字段名不在已知别名里，把表头发我加别名 |
| CSV 导入后全是 `other` | 没有自述列且没给 `--me`，补 `--me "你的昵称"` |
| 媒体消息被跳过 | 正常：训练/记忆只用文本；需要占位可加 `--keep-media`（JSON 导入） |
| 时间偏移 8 小时 | 导出工具给的是本地时间；导入会补时区，如需 UTC 自行后处理 |
| 文件 500MB+ 导入慢 | 先只导一个联系人；或先用 `--limit` 小样本验证（JSON 导入支持） |
| 想先验证流程 | `--dry-run` 只统计不写库 |
