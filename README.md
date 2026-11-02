# 微信自动回复机制

> 学习个人聊天风格（经历、故事、语气、表情包习惯），通过**屏幕截图 + OCR**读取消息上下文，用**键盘模拟**像真人一样回复，并根据消息长短模拟不同的打字耗时。

**非侵入式**：不 hook、不注入、不读数据库、不解密、不碰微信进程内存，与录屏/读屏软件同类操作。

详细技术方案见 [`微信自动回复技术实现方案.md`](./微信自动回复技术实现方案.md)。

---

## ⚠️ 免责声明

本项目**仅供学习交流**，请勿用于任何非法用途，否则后果自负。

- 请勿用于骚扰、诈骗、垃圾营销、群发广告等行为。
- 自动操作微信可能违反微信用户协议，存在**账号封禁风险**，使用前请自行评估，风险自担。
- 聊天记录含大量个人隐私，请在本机脱敏处理，**切勿上传第三方或提交到仓库**。
- 请勿处理转账、红包、收款等相关界面元素。

下载、使用本代码即表示你同意并承诺遵守以上条款。

---

## 快速开始（待代码落地后补充）

### 环境变量

复制 `.env.example` 为 `.env` 并填入真实值：

```bash
cp .env.example .env
```

`.env` 已被 `.gitignore` 排除，不会进入仓库。代码中通过环境变量读取密钥：

```python
import os
api_key = os.environ["LLM_API_KEY"]
```

在 Windows PowerShell 中临时设置环境变量：

```powershell
$env:LLM_API_KEY = "你的key"
$env:LLM_BASE_URL = "https://api.deepseek.com/v1"
$env:LLM_MODEL = "deepseek-chat"
```

### 安装依赖

```bash
pip install -r requirements.txt
```

（`requirements.txt` 与核心代码将在后续阶段补充。）

---

## 架构概览

```
离线（训练一次）   聊天记录导出 → 清洗脱敏 → 风格建模（LoRA 微调 / few-shot 样本）
在线（每条消息）   截屏 → 定位消息区 → OCR → 分"我/对方" → 去重 → LLM 生成 → 模拟打字 → 发送
```

---

## 参考项目

- [WeChat-DeepSeek-Auto-Response](https://github.com/LaoChouPro/WeChat-DeepSeek-Auto-Response)
- [polish-wechat-windows](https://github.com/Echosong/polish-wechat-windows)
- [WeClone](https://github.com/Hisou-Tenshi/weclone)
- [PyWxDump](https://github.com/xaoyaoo/PyWxDump)
- [HumanTyping](https://github.com/Lax3n/HumanTyping)

## License

[MIT](./LICENSE)
