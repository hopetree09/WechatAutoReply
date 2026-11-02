# 微信自动回复机制

> 学习个人聊天风格（经历、故事、语气、表情包习惯），通过**屏幕截图 + OCR**读取消息上下文，用**键盘模拟**像真人一样回复，并根据消息长短模拟不同的打字耗时。

**非侵入式**：不 hook、不注入、不读数据库、不解密、不碰微信进程内存，与录屏/读屏软件同类操作。

完整技术方案见 [`微信自动回复技术实现方案.md`](./微信自动回复技术实现方案.md)。

---

## ⚠️ 免责声明

本项目**仅供学习交流**，请勿用于任何非法用途，否则后果自负。

- 请勿用于骚扰、诈骗、垃圾营销、群发广告等行为。
- 自动操作微信可能违反微信用户协议，存在**账号封禁风险**，使用前请自行评估，风险自担。
- 聊天记录含大量个人隐私，请在本机脱敏处理，**切勿上传第三方或提交到仓库**。
- 请勿处理转账、红包、收款等相关界面元素。

下载、使用本代码即表示你同意并承诺遵守以上条款。

---

## 功能特性

- 📷 屏幕截图（DDA/WGC）+ 离线 OCR 读取消息上下文
- 🎨 按气泡颜色区分「我 / 对方」
- 🧠 个人风格：自述画像 + few-shot 口吻样本（进阶可 LoRA 微调，见方案文档）
- ⌨️ 拟人打字：打字速度 ∝ 消息长度，带随机抖动与分片停顿
- 🔒 安全开关：默认**只打印不发送**，确认无误再开自动发送

---

## 目录结构

```
wechat-autoreply/
├── main.py                 # 入口：主循环
├── config.example.py       # 配置模板（复制为 config.py 使用）
├── requirements.txt
├── core/
│   ├── capture.py          # 截图（DDA/GDI）
│   ├── ocr.py              # RapidOCR 封装
│   ├── parser.py           # 消息区定位 + 分说话人
│   ├── state.py            # 会话状态 / 去重
│   ├── generator.py        # LLM 生成（风格 prompt + few-shot）
│   └── typist.py           # 打字速度模型 + 键盘执行
├── scripts/
│   └── calibrate.py        # 标定：窗口坐标 / 气泡颜色
└── data/
    ├── profile.md.example  # 自述画像模板
    └── fewshot/            # 口吻样本目录（*.txt 不进仓库）
```

---

## 快速开始（换新电脑后）

### 0. 克隆

```bash
git clone https://github.com/hopetree09/WechatAutoReply.git
cd WechatAutoReply
```

### 1. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt
```

> 首次运行 `RapidOCR` 会自动下载 OCR 模型，需联网。

### 2. 配置

```bash
copy config.example.py config.py
copy .env.example .env
```

编辑 `.env`，填入你的 API key（`.env` 已被 `.gitignore` 忽略，不会进仓库）：

```ini
LLM_API_KEY=你的key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

或者用环境变量临时设置（PowerShell）：

```powershell
$env:LLM_API_KEY = "你的key"
```

### 3. 标定（每台新电脑 / 微信改版后都要做）

```bash
python scripts/calibrate.py
```

按提示依次点击：**自己的气泡 → 对方的气泡 → 输入框**，把打印结果填进 `config.py`。

### 4. 试运行（默认只打印，不发送）

`config.py` 里 `AUTO_SEND` 默认是 `False`，先跑：

```bash
python main.py
```

看到对方消息和回复都正常打印后，再把 `AUTO_SEND` 改成 `True` 开启自动发送。

---

## 配置说明（config.py）

| 配置项 | 说明 |
|---|---|
| `CAPTURE_BACKEND` | `dxcam`（推荐）或 `mss`（截图黑屏时切换） |
| `WECHAT_WINDOW_CLASS` | 微信窗口类名（用 calibrate.py 探测） |
| `MY_BUBBLE_RGB` / `OTHER_BUBBLE_RGB` | 自己/对方气泡颜色（用 calibrate.py 采样） |
| `TYPING_PARAMS.cps_base` | 每字秒数，越大打字越慢（0.25~0.6） |
| `SEND_KEY` | `enter` 或 `ctrl+enter` |
| `AUTO_SEND` | `False` 只打印；`True` 自动发送 |

---

## 参考项目

- [WeChat-DeepSeek-Auto-Response](https://github.com/LaoChouPro/WeChat-DeepSeek-Auto-Response)
- [polish-wechat-windows](https://github.com/Echosong/polish-wechat-windows)
- [WeClone](https://github.com/Hisou-Tenshi/weclone)
- [PyWxDump](https://github.com/xaoyaoo/PyWxDump)
- [HumanTyping](https://github.com/Lax3n/HumanTyping)

## License

[MIT](./LICENSE)
