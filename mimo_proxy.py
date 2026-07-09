"""
MiMo Reasoning Content Proxy v1.6
==================================
v1.3: 当缓存未命中时，剥离 assistant 消息的 tool_calls（降级为纯文本），
     避免 400 错误。MiMo 只对有 tool_calls 的 assistant 消息要求 reasoning_content。
v1.4: 修复非流式模式下上游返回错误时的处理：检查状态码、添加重试逻辑、
     确保不会返回空 content。
v1.5: 修复 v1.3 降级逻辑导致模型"用文本模仿调用工具"的污染问题。
     缓存未命中时不再剥离 tool_calls，改为注入占位符 reasoning_content。
v1.6: 增加历史污染清理：v1.3 在 content 末尾追加的 "[Called X]" 伪工具调用文本
     会被客户端永久保存到对话历史里，导致模型反复模仿生成假调用。每次请求时
     主动扫描所有 assistant 消息，从 content 末尾剥掉这种污染标记。
"""

import re

# 缓存未命中时使用的占位符 reasoning_content
PLACEHOLDER_REASONING = (
    "(reasoning_content was not preserved by the client; this is a placeholder "
    "to satisfy the MiMo API contract. Continue based on visible content and tool_calls.)"
)

# 匹配 v1.3 / v1.4 旧版本在 content 末尾追加的伪工具调用标记
# 例: " [Called WebFetch]"、" [Called WebFetch] [Called RunCommand]"
_FAKE_TOOL_CALL_SUFFIX = re.compile(r"(?:\s*\[Called\s+[^\]]+\])+\s*$")

import hashlib
import json
import logging
import time
from collections import OrderedDict
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

# ─── 配置 ──────────────────────────────────────────────────────
MIMO_API_BASE = "https://token-plan-cn.xiaomimimo.com/v1"
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8899
CACHE_MAX_SIZE = 2000
CACHE_TTL = 7200

log = logging.getLogger("mimo-proxy")

# ─── 缓存 ──────────────────────────────────────────────────────
_cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
_tool_call_index: dict[str, str] = {}
_http_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(300, connect=30),
            follow_redirects=True,
        )
    return _http_client


def _msg_hash(msg: dict) -> str:
    content = msg.get("content") or ""
    tool_calls = json.dumps(msg.get("tool_calls") or [], sort_keys=True, ensure_ascii=False)
    raw = f"{content}||{tool_calls}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _extract_tool_call_ids(msg: dict) -> list[str]:
    return [tc.get("id", "") for tc in msg.get("tool_calls") or [] if tc.get("id")]


def _cache_get(key: str) -> str | None:
    if key in _cache:
        val, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            _cache.move_to_end(key)
            return val
        del _cache[key]
    return None


def _cache_set(key: str, value: str):
    if key in _cache:
        del _cache[key]
    _cache[key] = (value, time.time())
    while len(_cache) > CACHE_MAX_SIZE:
        _cache.popitem(last=False)


def _cache_set_with_index(key: str, value: str, tool_call_ids: list[str]):
    _cache_set(key, value)
    for tid in tool_call_ids:
        _tool_call_index[tid] = value


def _find_by_tool_call_ids(msg: dict) -> str | None:
    for tid in _extract_tool_call_ids(msg):
        if tid in _tool_call_index:
            return _tool_call_index[tid]
    return None


# ─── 核心逻辑 ──────────────────────────────────────────────────

def _scrub_fake_tool_call_markers(messages: list[dict]) -> int:
    """
    清理历史里 v1.3/v1.4 旧版代理留下的伪工具调用标记。
    那些版本会在 content 末尾追加 "[Called X]"，客户端会把它持久化进对话历史，
    模型从历史里学到这种模式后会持续模仿，输出 "让我调用 X [Called X]" 然后停止。
    每次请求时把这种后缀剥掉，避免污染传递给 MiMo。
    """
    scrubbed = 0
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not content or not isinstance(content, str):
            continue
        cleaned = _FAKE_TOOL_CALL_SUFFIX.sub("", content)
        if cleaned != content:
            msg["content"] = cleaned
            scrubbed += 1
            log.info("🧹 Scrubbed fake [Called ...] markers from msg[%d]", i)
    return scrubbed


def inject_reasoning(messages: list[dict]) -> tuple[int, int]:
    """
    处理 assistant 消息：
    1. 有缓存 → 注入真实的 reasoning_content
    2. 无缓存 → 注入占位符 reasoning_content（保留 tool_calls 结构）

    返回 (注入数, 占位符注入数)
    """
    injected = 0
    placeholder = 0

    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        if not msg.get("tool_calls"):
            continue
        if msg.get("reasoning_content"):
            continue

        # 尝试查找缓存
        h = _msg_hash(msg)
        cached = _cache_get(h)
        if not cached:
            cached = _find_by_tool_call_ids(msg)

        if cached:
            # ✅ 有缓存，注入真实内容
            msg["reasoning_content"] = cached
            injected += 1
            log.info("✅ Injected reasoning_content into msg[%d] [%s] (%d chars)", i, h[:8], len(cached))
        else:
            # ⚠️ 无缓存，注入占位符（保留 tool_calls 结构，避免污染对话）
            tc_ids = _extract_tool_call_ids(msg)
            log.warning("⚠️  No cache for msg[%d] [%s] tool_call_ids=%s → injecting placeholder",
                        i, h[:8], tc_ids)
            msg["reasoning_content"] = PLACEHOLDER_REASONING
            placeholder += 1

    return injected, placeholder


def cache_reasoning_from_message(msg: dict):
    rc = msg.get("reasoning_content")
    if rc and msg.get("tool_calls"):
        h = _msg_hash(msg)
        tc_ids = _extract_tool_call_ids(msg)
        _cache_set_with_index(h, rc, tc_ids)
        log.info("📦 Cached reasoning [%s] (%d chars) tc_ids=%s", h[:8], len(rc), tc_ids)


# ─── SSE 流式处理 ──────────────────────────────────────────────

def _sse(data: str) -> bytes:
    return f"data: {data}\n\n".encode("utf-8")


async def _stream_proxy(client: httpx.AsyncClient, url: str, headers: dict, body: dict):
    acc_content = ""
    acc_reasoning = ""
    acc_tool_calls: list[dict] = []

    # 流式模式也加重试（针对 5xx）
    last_error = None
    for attempt in range(3):
        try:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    error_text = error_body.decode("utf-8", errors="replace")
                    log.warning("⚠️ Stream upstream %d (attempt %d): %s", resp.status_code, attempt + 1, error_text[:200])
                    if resp.status_code < 500:
                        yield _sse(error_text)
                        return
                    last_error = error_text
                    if attempt < 2:
                        import asyncio
                        await asyncio.sleep(1 * (attempt + 1))
                        continue
                    yield _sse(json.dumps({"error": {"message": f"MiMo API error after retries: {last_error[:200]}", "code": "502"}}))
                    return

                buffer = ""
                async for raw_chunk in resp.aiter_bytes():
                    buffer += raw_chunk.decode("utf-8", errors="replace")

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.rstrip("\r")

                        if line.startswith("data: "):
                            payload = line[6:].strip()

                            if payload == "[DONE]":
                                if acc_reasoning and (acc_content or acc_tool_calls):
                                    synthetic = {
                                        "role": "assistant",
                                        "content": acc_content,
                                        "tool_calls": acc_tool_calls,
                                        "reasoning_content": acc_reasoning,
                                    }
                                    h = _msg_hash(synthetic)
                                    tc_ids = _extract_tool_call_ids(synthetic)
                                    _cache_set_with_index(h, acc_reasoning, tc_ids)
                                    log.info("📦 Cached streaming reasoning [%s] (%d chars)", h[:8], len(acc_reasoning))
                                yield _sse("[DONE]")
                                continue

                            try:
                                chunk = json.loads(payload)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                rc = delta.get("reasoning_content")
                                if rc:
                                    acc_reasoning += rc
                                c = delta.get("content")
                                if c:
                                    acc_content += c
                                for tc in delta.get("tool_calls") or []:
                                    idx = tc.get("index", 0)
                                    while len(acc_tool_calls) <= idx:
                                        acc_tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                                    if tc.get("id"):
                                        acc_tool_calls[idx]["id"] = tc["id"]
                                    fn = tc.get("function", {})
                                    if fn.get("name"):
                                        acc_tool_calls[idx]["function"]["name"] += fn["name"]
                                    if fn.get("arguments"):
                                        acc_tool_calls[idx]["function"]["arguments"] += fn["arguments"]
                            except (json.JSONDecodeError, IndexError, KeyError):
                                pass

                            yield _sse(payload)

                        elif line.strip() == "":
                            yield b"\n"
                        elif line.startswith(":"):
                            yield (line + "\n\n").encode("utf-8")
                        else:
                            yield (line + "\n").encode("utf-8")
                # 流成功完成，退出重试循环
                return

        except httpx.TimeoutException as e:
            log.warning("⚠️ Stream timeout (attempt %d): %s", attempt + 1, e)
            last_error = str(e)
            if attempt < 2:
                import asyncio
                await asyncio.sleep(2 * (attempt + 1))
                continue
        except Exception as e:
            log.error("❌ Stream error: %s", e, exc_info=True)
            yield _sse(json.dumps({"error": f"Proxy error: {e}"}))
            return

    # 所有重试都失败
    yield _sse(json.dumps({"error": {"message": f"Stream error after retries: {last_error}", "code": "502"}}))


# ─── HTTP 端点 ─────────────────────────────────────────────────

async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    messages = body.get("messages", [])
    scrubbed = _scrub_fake_tool_call_markers(messages)
    injected, placeholder = inject_reasoning(messages)
    if injected or placeholder or scrubbed:
        log.info("🔧 Scrubbed=%d, Injected=%d, Placeholder=%d", scrubbed, injected, placeholder)

    headers = {}
    auth = request.headers.get("authorization")
    if auth:
        headers["authorization"] = auth

    is_stream = body.get("stream", False)
    upstream = f"{MIMO_API_BASE}/chat/completions"
    client = _get_client()

    if is_stream:
        return StreamingResponse(
            _stream_proxy(client, upstream, headers, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )
    else:
        # 非流式：重试逻辑处理 MiMo 偶发 500
        last_error = None
        for attempt in range(3):
            try:
                resp = await client.post(upstream, headers=headers, json=body)
                if resp.status_code != 200:
                    error_text = resp.text
                    log.warning("⚠️ Upstream %d (attempt %d): %s", resp.status_code, attempt + 1, error_text[:200])
                    # 4xx 错误不重试（参数问题），5xx 重试
                    if resp.status_code < 500:
                        return JSONResponse(
                            {"error": {"message": f"Upstream error: {error_text[:200]}", "code": str(resp.status_code)}},
                            status_code=resp.status_code,
                        )
                    last_error = error_text
                    if attempt < 2:
                        import asyncio
                        await asyncio.sleep(1 * (attempt + 1))
                        continue
                    # 最后一次仍然失败
                    return JSONResponse(
                        {"error": {"message": f"MiMo API error after 3 attempts: {last_error[:200]}", "code": "502"}},
                        status_code=502,
                    )

                data = resp.json()

                # 检查返回数据是否有效
                choices = data.get("choices", [])
                if not choices:
                    log.warning("⚠️ Empty choices in response")
                    return JSONResponse(
                        {"error": {"message": "MiMo API returned empty choices", "code": "502"}},
                        status_code=502,
                    )

                # 检查 content 是否为空（MiMo 有时只返回 reasoning_content）
                msg = choices[0].get("message", {})
                if not msg.get("content") and not msg.get("tool_calls") and msg.get("reasoning_content"):
                    log.warning("⚠️ Response has reasoning_content but no content, setting fallback")
                    msg["content"] = msg["reasoning_content"]

                for choice in choices:
                    cache_reasoning_from_message(choice.get("message", {}))

                return JSONResponse(content=data, status_code=200)

            except httpx.TimeoutException as e:
                log.warning("⚠️ Timeout (attempt %d): %s", attempt + 1, e)
                last_error = str(e)
                if attempt < 2:
                    import asyncio
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
            except Exception as e:
                log.error("❌ Error: %s", e, exc_info=True)
                return JSONResponse({"error": {"message": str(e), "code": "500"}}, status_code=500)

        return JSONResponse(
            {"error": {"message": f"Proxy error after retries: {last_error}", "code": "502"}},
            status_code=502,
        )


async def list_models(request: Request):
    headers = {}
    auth = request.headers.get("authorization")
    if auth:
        headers["authorization"] = auth
    client = _get_client()
    try:
        resp = await client.get(f"{MIMO_API_BASE}/models", headers=headers)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)


async def root(request: Request):
    return JSONResponse({
        "status": "running",
        "service": "MiMo Reasoning Content Proxy v1.6",
        "cache_size": len(_cache),
        "tool_call_index_size": len(_tool_call_index),
        "upstream": MIMO_API_BASE,
    })


async def health(request: Request):
    return JSONResponse({"ok": True})


@asynccontextmanager
async def lifespan(app):
    global _http_client
    _http_client = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=30), follow_redirects=True)
    log.info("🚀 httpx client initialized")
    yield
    if _http_client:
        await _http_client.aclose()


routes = [
    Route("/", root),
    Route("/health", health),
    Route("/v1/models", list_models),
    Route("/models", list_models),
    Route("/v1/chat/completions", chat_completions, methods=["POST"]),
    Route("/chat/completions", chat_completions, methods=["POST"]),
]

app = Starlette(routes=routes, lifespan=lifespan)

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    log.info("🚀 MiMo Proxy v1.6 on %s:%d → %s", LISTEN_HOST, LISTEN_PORT, MIMO_API_BASE)
    
    # 显示正确的 Trae 配置地址
    import socket
    local_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass
    
    print()
    print("=" * 60)
    print("  ✅ 代理已启动！请在 Trae 中配置以下地址：")
    print()
    print(f"  本机访问: http://127.0.0.1:{LISTEN_PORT}/v1/chat/completions")
    print(f"  局域网:   http://{local_ip}:{LISTEN_PORT}/v1/chat/completions")
    print()
    print("  ⚠️  注意：")
    print("  1. 地址必须是完整路径，包含 /v1/chat/completions")
    print("  2. 不要用 0.0.0.0，那是监听地址不是访问地址")
    print("  3. API Key 填你的 MiMo API Key")
    print("=" * 60)
    print()
    
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT, log_level="info")
