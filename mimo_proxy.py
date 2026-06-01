"""
MiMo Reasoning Content Proxy v1.6
==================================
v1.3: 当缓存未命中时，剥离 assistant 消息的 tool_calls（降级为纯文本），
     避免 400 错误。MiMo 只对有 tool_calls 的 assistant 消息要求 reasoning_content。
v1.4: 修复非流式模式下上游返回错误时的处理：检查状态码、添加重试逻辑、
     确保不会返回空 content。
v1.5: 修复工具链断裂问题：缓存未命中时不再剥离 tool_calls（会导致后续 tool 消息
     成为孤儿），改为注入占位 reasoning_content 保持工具链完整。
v1.6: 缓存预热 + 索引增强 + 429限流重试 + 并发控制。
"""

import asyncio
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
CACHE_MAX_SIZE = 5000
CACHE_TTL = 43200  # 12小时，子代理长时间使用也不易过期

log = logging.getLogger("mimo-proxy")

# ─── 缓存 ──────────────────────────────────────────────────────
_cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
_tool_call_index: dict[str, str] = {}
_process_lock = asyncio.Lock()  # 串行处理锁：一次一个请求，最大化缓存复用并避免限流
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

def _warm_cache_from_messages(messages: list[dict]):
    """
    预热缓存：遍历 messages，将已包含 reasoning_content 的 assistant 消息
    写入缓存。这样同一请求中的后续消息（或并发子代理请求）可以通过
    tool_call_id 共享缓存。
    """
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        rc = msg.get("reasoning_content")
        if not rc or not msg.get("tool_calls"):
            continue
        h = _msg_hash(msg)
        tc_ids = _extract_tool_call_ids(msg)
        _cache_set_with_index(h, rc, tc_ids)


def inject_reasoning(messages: list[dict]) -> tuple[int, int]:
    """
    处理 assistant 消息：
    1. 有缓存 → 注入 reasoning_content
    2. 无缓存 → 注入占位 reasoning_content（保持工具链完整）

    返回 (注入数, 占位数)
    """

    # 第一步：预热缓存
    # 子代理可能并发请求，提前将已有 reasoning_content 的消息缓存起来，
    # 供同批次其他消息或其他子代理的请求复用。
    _warm_cache_from_messages(messages)

    injected = 0
    degraded = 0

    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        if not msg.get("tool_calls"):
            continue
        if msg.get("reasoning_content"):
            continue

        h = _msg_hash(msg)
        cached = _cache_get(h)
        if not cached:
            cached = _find_by_tool_call_ids(msg)
            if cached:
                # 通过 tool_call_id 命中缓存，也写入主缓存
                # 后续同 hash 的消息可直接命中主缓存（更快）
                _cache_set(h, cached)

        if cached:
            msg["reasoning_content"] = cached
            injected += 1
            # 同步更新 tool_call_index
            # 这样同请求中后续相同 tool_call_id 的消息，
            # 或其他并发子代理的消息，也能通过索引找到
            tc_ids = _extract_tool_call_ids(msg)
            for tid in tc_ids:
                _tool_call_index[tid] = cached
            log.info("✅ Injected reasoning_content into msg[%d] [%s] (%d chars)", i, h[:8], len(cached))
        else:
            tc_ids = _extract_tool_call_ids(msg)
            log.warning("⚠️  No cache for msg[%d] [%s] tool_call_ids=%s → injecting placeholder reasoning",
                        i, h[:8], tc_ids)

            placeholder = "(reasoning not cached for this message)"
            msg["reasoning_content"] = placeholder
            degraded += 1

    return injected, degraded


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

    last_error = None
    for attempt in range(3):
        try:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    error_text = error_body.decode("utf-8", errors="replace")
                    log.warning("⚠️ Stream upstream %d (attempt %d): %s", resp.status_code, attempt + 1, error_text[:200])
                    if resp.status_code == 429:
                        last_error = error_text
                        if attempt < 2:
                            wait = 2 ** (attempt + 1)  # 2s, 4s
                            log.warning("⏳ Rate limited, retrying in %ds (attempt %d/3)", wait, attempt + 1)
                            await asyncio.sleep(wait)
                            continue
                        yield _sse(json.dumps({"error": {"message": f"MiMo API rate limited after retries: {last_error[:200]}", "code": "429"}}))
                        return
                    if resp.status_code < 500:
                        yield _sse(error_text)
                        return
                    last_error = error_text
                    if attempt < 2:
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
                return

        except httpx.TimeoutException as e:
            log.warning("⚠️ Stream timeout (attempt %d): %s", attempt + 1, e)
            last_error = str(e)
            if attempt < 2:
                await asyncio.sleep(2 * (attempt + 1))
                continue
        except Exception as e:
            log.error("❌ Stream error: %s", e, exc_info=True)
            yield _sse(json.dumps({"error": f"Proxy error: {e}"}))
            return

    yield _sse(json.dumps({"error": {"message": f"Stream error after retries: {last_error}", "code": "502"}}))


# ─── HTTP 端点 ─────────────────────────────────────────────────

async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    headers = {}
    auth = request.headers.get("authorization")
    if auth:
        headers["authorization"] = auth

    is_stream = body.get("stream", False)
    upstream = f"{MIMO_API_BASE}/chat/completions"
    client = _get_client()

    if is_stream:
        # 流式响应：在异步生成器内部持有锁，串行处理
        # 这样可以确保上一个请求刚完成的缓存立刻被下一个请求利用
        # 同时避免并发请求触发上游限流
        async def locked_stream():
            async with _process_lock:
                messages = body.get("messages", [])
                injected, degraded = inject_reasoning(messages)
                if injected or degraded:
                    log.info("🔧 Injected=%d, Placeholder=%d", injected, degraded)
                async for chunk in _stream_proxy(client, upstream, headers, body):
                    yield chunk

        return StreamingResponse(
            locked_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    # 非流式：直接在锁内处理
    async with _process_lock:
        messages = body.get("messages", [])
        injected, degraded = inject_reasoning(messages)
        if injected or degraded:
            log.info("🔧 Injected=%d, Placeholder=%d", injected, degraded)

        last_error = None
        for attempt in range(3):
            try:
                resp = await client.post(upstream, headers=headers, json=body)
                if resp.status_code != 200:
                    error_text = resp.text
                    log.warning("⚠️ Upstream %d (attempt %d): %s", resp.status_code, attempt + 1, error_text[:200])
                    if resp.status_code == 429:
                        last_error = error_text
                        if attempt < 2:
                            wait = 2 ** (attempt + 1)
                            log.warning("⏳ Rate limited, retrying in %ds (attempt %d/3)", wait, attempt + 1)
                            await asyncio.sleep(wait)
                            continue
                        return JSONResponse(
                            {"error": {"message": f"MiMo API rate limited after retries: {last_error[:200]}", "code": "429"}},
                            status_code=429,
                        )
                    if resp.status_code < 500:
                        return JSONResponse(
                            {"error": {"message": f"Upstream error: {error_text[:200]}", "code": str(resp.status_code)}},
                            status_code=resp.status_code,
                        )
                    last_error = error_text
                    if attempt < 2:
                        await asyncio.sleep(1 * (attempt + 1))
                        continue
                    return JSONResponse(
                        {"error": {"message": f"MiMo API error after 3 attempts: {last_error[:200]}", "code": "502"}},
                        status_code=502,
                    )

                data = resp.json()

                choices = data.get("choices", [])
                if not choices:
                    log.warning("⚠️ Empty choices in response")
                    return JSONResponse(
                        {"error": {"message": "MiMo API returned empty choices", "code": "502"}},
                        status_code=502,
                    )

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
