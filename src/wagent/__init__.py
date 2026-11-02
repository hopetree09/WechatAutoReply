"""wagent：桌面微信 GUI Agent。

模块划分（对应技术方案的四层）：
    perception  感知层：窗口定位 / 截图 / OCR / 消息结构化 / 输入框定位
    im          IM 适配层：把「屏幕」抽象成可替换的聊天通道
    llm         模型层：本地 / 云端双通道
    memory      记忆层：L0 工作记忆 ~ L5 风格样本库
    agent       决策层：工具调用 + 主循环
    guard       安全层：注入检测 / PII / 发送前校验
    eval        评测层
"""

__version__ = "0.1.0"
