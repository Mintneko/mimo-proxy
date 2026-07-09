# MiMo Reasoning Content Proxy

解决小米 MiMo API 强制要求回传 `reasoning_content` 字段导致 Trae、Cursor 等客户端出现 **400 Param Incorrect** 报错的轻量级代理中间件。

## 问题背景

2026年5月12日，小米 MiMo API 开放平台发布协议变更：在 Agent 类产品的多轮会话中，如果开启思考模式（Thinking Mode）且历史消息包含工具调用（tool_calls），assistant 消息必须完整回传 `reasoning_content` 字段，否则 API 返回 400 错误。

```
HTTP/1.1 400 Bad Request
{
  "error": {
    "message": "Param Incorrect",
    "param": "The reasoning_content in the thinking mode must be passed back to the API.",
    "code": "400"
  }
}
```

受影响的客户端包括：Trae、Cursor、GitHub Copilot CLI、Roo Code、Codex、Zed、AutoGen 等。

## 解决方案

本代理作为 Trae 与 MiMo API 之间的中间层：

```
Trae → MiMo Reasoning Proxy → MiMo API
         ↓ 拦截响应，缓存 reasoning_content
         ↓ 下次请求自动注入回 assistant 消息
```

核心逻辑：
1. **拦截响应**：从 MiMo 返回的 assistant 消息中提取 `reasoning_content`，按 `content + tool_calls` 哈希缓存
2. **注入请求**：当 Trae 发送后续请求时，为缺少 `reasoning_content` 的 assistant 消息自动注入缓存值
3. **占位符兜底**：缓存未命中时（例如代理启动前已存在的旧对话），保留原始 `tool_calls` 结构，仅注入占位符 `reasoning_content` 满足 API 字段校验
4. **历史清洗**：早期版本（v1.3 / v1.4）会把 `tool_calls` 改写成 `[Called X]` 文本注入 `content`，这种污染会被客户端持久化保存，导致模型反复模仿生成假调用。每次请求时主动剥掉 content 末尾的 `[Called ...]` 后缀

## 快速开始

### 安装依赖

```bash
pip install fastapi uvicorn httpx
```

### 启动代理

```bash
python mimo_proxy.py
```

默认监听 `0.0.0.0:8899`，上游指向 Token Plan API。

### 配置 Trae

1. 打开 Trae → 设置 → Models → 你的 MiMo 自定义模型
2. 将 **Custom Request URL** 改为：

```
http://127.0.0.1:8899/v1/chat/completions
```

> ⚠️ **常见错误：**
> - ❌ `http://0.0.0.0:8899/v1` — `0.0.0.0` 是监听地址，不能用来访问
> - ❌ `https://127.0.0.1:8899/v1` — https错误的，应写http
> - ✅ `http://127.0.0.1:8899/v1` — 非完整路径
> - ✅ `http://127.0.0.1:8899/v1/chat/completions` — 正确格式
> 如果代理部署在其他机器上，将 `127.0.0.1` 替换为该机器的 IP 地址。

## 配置

编辑 `mimo_proxy.py` 顶部的常量：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MIMO_API_BASE` | `https://token-plan-cn.xiaomimimo.com/v1` | MiMo API 地址（Token Plan） |
| `LISTEN_HOST` | `0.0.0.0` | 监听地址 |
| `LISTEN_PORT` | `8899` | 监听端口 |
| `CACHE_MAX_SIZE` | `2000` | 最大缓存条目数 |
| `CACHE_TTL` | `7200` | 缓存过期时间（秒） |

如果使用按量付费 API，将 `MIMO_API_BASE` 改为：
```
https://api.xiaomimimo.com/v1
```

## Systemd 服务（可选）

```bash
# 创建服务文件
cat > /etc/systemd/system/mimo-proxy.service << 'EOF'
[Unit]
Description=MiMo Reasoning Content Proxy
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/mimo-proxy
ExecStart=/usr/bin/python3 /opt/mimo-proxy/mimo_proxy.py
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# 启动
systemctl daemon-reload
systemctl enable --now mimo-proxy

# 查看日志
journalctl -u mimo-proxy -f
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 状态页（含缓存大小） |
| `/health` | GET | 健康检查 |
| `/v1/models` | GET | 代理 MiMo 模型列表 |
| `/v1/chat/completions` | POST | 代理聊天补全（支持流式） |
| `/chat/completions` | POST | 同上（兼容路径） |

## 工作原理

```
┌─────────┐     POST /v1/chat/completions     ┌──────────┐
│  Trae   │ ──────────────────────────────────→│  Proxy   │
│         │                                     │          │
│         │  1. 检查 assistant 消息              │          │
│         │     有 tool_calls 但无               │          │
│         │     reasoning_content?              │          │
│         │                                     │          │
│         │  2a. 有缓存 → 注入                   │          │
│         │  2b. 无缓存 → 剥离 tool_calls        │          │
│         │                                     │          │
│         │     ─────────────────────────────→  │  MiMo    │
│         │                                     │  API     │
│         │  3. 缓存响应中的 reasoning_content   │          │
│         │ ←─────────────────────────────────  │          │
│         │                                     │          │
└─────────┘                                     └──────────┘
```

## 已知限制

- 缓存基于内存，重启后丢失（新对话会自动重建）
- 降级处理（剥离 tool_calls）会导致模型丢失工具调用的上下文
- 仅支持 OpenAI 兼容的 `/v1/chat/completions` 端点

## 邀请码
我在用 MiMo 开放平台体验 小米顶尖模型 MiMo V2.5等 ，通过我的邀请码注册为新用户，即得 ¥10 API 体验金。邀请码：B8DMC5。注册：https://platform.xiaomimimo.com?ref=B8DMC5（注册后点控制台左下方入口填入，体验金40天有效）


## 相关链接

- [小米 MiMo API 官方公告](https://platform.xiaomimimo.com/docs/zh-CN/usage-guide/passing-back-reasoning_content)
- [LINUX DO 讨论帖](https://linux.do/t/topic/2165444)
- [Trae 论坛反馈](https://forum.trae.cn/t/topic/17335)

## License

MIT
