"""OpenAI-compatible HTTP server (greedy decoding, one request at a time).
  GET  /v1/models
  POST /v1/chat/completions  {messages, tools, max_tokens, stream, chat_template_kwargs: {enable_thinking}}
Tool calls: `tools` are passed to the original chat template; the model's tool-call syntax is converted to OpenAI
`tool_calls` with JSON arguments. Sampling: temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty
and seed are honoured; without temperature (or temperature 0) decoding is greedy."""
from __future__ import annotations

import base64
import io
import json
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _image(url):
    from PIL import Image
    if url.startswith("data:"):
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    raise ValueError("only data: URLs are supported for images")


def to_template(messages, images):
    """OpenAI messages -> template messages. Image parts are collected into `images` (in order)."""
    def text_of(content):
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        parts = []
        for p in content:
            t = p.get("type")
            if t in ("text", "input_text"):
                parts.append({"type": "text", "text": p.get("text", "")})
            elif t in ("image_url", "image", "input_image"):
                u = p.get("image_url"); u = u.get("url") if isinstance(u, dict) else u
                images.append(_image(u or p.get("url", ""))); parts.append({"type": "image"})
        return parts if any(x["type"] == "image" for x in parts) else "".join(x["text"] for x in parts)

    out = []
    for x in messages:
        r = x["role"]
        if r == "tool":
            out.append({"role": "tool", "tool_call_id": x.get("tool_call_id"), "name": x.get("name"), "content": text_of(x.get("content"))})
            continue
        y = {"role": "system" if r == "developer" else r, "content": text_of(x.get("content"))}
        if r == "assistant":
            if x.get("reasoning_content"):
                y["reasoning_content"] = x["reasoning_content"]
            if x.get("tool_calls"):
                tcs = []
                for tc in x["tool_calls"]:
                    f = tc.get("function", {}); a = f.get("arguments")
                    if isinstance(a, str):
                        try:
                            a = json.loads(a) if a.strip() else {}
                        except json.JSONDecodeError:
                            a = {"raw": a}
                    tcs.append({"id": tc.get("id"), "type": "function", "function": {"name": f.get("name"), "arguments": a if isinstance(a, dict) else {"value": a}}})
                y["tool_calls"] = tcs
        out.append(y)
    return out


def prepare(eng, chat, messages, thinking=False, tools=None):
    """-> (ids, generate kwargs) including image expansion."""
    images = []
    msgs = to_template(messages, images)
    ids, _ = chat.encode(msgs, thinking=thinking, tools=tools)
    if not images:
        return ids, {}
    if not hasattr(eng, "encode_images"):
        raise ValueError("this model has no vision encoder")
    soft, ns = eng.encode_images(images)
    ids, spans = chat.expand_images(ids, ns)
    return ids, {"spans": spans, "soft": soft}


def serve(eng, chat, host, port, name):
    print(f"ready: {name} at http://{host}:{port}/v1 (context {eng.ctx} tokens)", flush=True)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "xeq", "context_length": eng.ctx}]})
            elif self.path.rstrip("/") in ("", "/health"):
                self._json(200, {"status": "ok"})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                return self._json(404, {"error": {"message": "not found"}})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                ck = req.get("chat_template_kwargs") or {}
                think = bool(ck.get("enable_thinking", req.get("enable_thinking", False)))
                tools = [t for t in (req.get("tools") or []) if t.get("type", "function") == "function"]
                if req.get("tool_choice") == "none":
                    tools = []
                ids, kw = prepare(eng, chat, req["messages"], think, tools)
                mx = int(req.get("max_tokens") or req.get("max_completion_tokens") or 4096)
                from .sampling import options
                samp = options(req)
            except Exception as e:
                return self._json(400, {"error": {"message": f"bad request: {e}"}})
            cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
            stream = bool(req.get("stream"))
            print(f"[{datetime.now():%H:%M:%S}] {cid[-6:]}: {len(req['messages'])} messages, {len(tools)} tools, prompt {len(ids)} tokens, "
                  f"max {mx}, thinking {'on' if think else 'off'}, stream {stream}" + (f", sampling {samp}" if samp else ""), flush=True)

            def finish(out):
                r, c, calls = chat.split(chat.decode(out), think)
                tcs = [{"id": "call_" + uuid.uuid4().hex[:20], "type": "function", "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}} for n, a in calls]
                return r, c, tcs

            def log(st, tcs):
                tp, tg, n, re_n = st
                print(f"   done: prompt {re_n} tokens {tp:.2f}s, {n} tokens {max(n - 1, 1) / max(tg, 1e-9):.1f} tok/s"
                      + (f", tool calls {[t['function']['name'] for t in tcs]}" if tcs else ""), flush=True)
            if not stream:
                try:
                    out, fin, st = eng.generate(ids, mx, reuse=req.get("kv_reuse", True), sampling=samp, **kw)
                except ValueError as e:
                    return self._json(400, {"error": {"message": str(e)}})
                r, c, tcs = finish(out)
                msg = {"role": "assistant", "content": c if (c or not tcs) else None}
                if r:
                    msg["reasoning_content"] = r
                if tcs:
                    msg["tool_calls"] = tcs; fin = "tool_calls"
                log(st, tcs)
                return self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": name,
                                        "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
                                        "usage": {"prompt_tokens": len(ids), "completion_tokens": st[2], "total_tokens": len(ids) + st[2]}})
            self.send_response(200); self.send_header("Content-Type", "text/event-stream; charset=utf-8"); self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close"); self.end_headers()
            sent = {"r": "", "c": ""}

            def ev(delta, fin=None, usage=None):
                ch = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": name, "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
                if usage:
                    ch["usage"] = usage
                self.wfile.write(("data: " + json.dumps(ch, ensure_ascii=False) + "\n\n").encode("utf-8")); self.wfile.flush()
            ev({"role": "assistant", "content": ""})

            def push(toks):
                txt = chat.decode(toks)
                if txt.endswith("�"):
                    return
                r, c = chat.visible(txt, think)
                for key, v, field in (("r", r, "reasoning_content"), ("c", c, "content")):
                    if v.startswith(sent[key]) and len(v) > len(sent[key]):
                        ev({field: v[len(sent[key]):]}); sent[key] = v
            try:
                out, fin, st = eng.generate(ids, mx, on_tokens=push, reuse=req.get("kv_reuse", True), sampling=samp, **kw)
                r, c, tcs = finish(out)
                if not tcs:
                    push(out)
                else:
                    ev({"tool_calls": [dict(index=i, **t) for i, t in enumerate(tcs)]}); fin = "tool_calls"
                us = {"prompt_tokens": len(ids), "completion_tokens": st[2], "total_tokens": len(ids) + st[2]}
                ev({}, fin, us if (req.get("stream_options") or {}).get("include_usage") else None)
                self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
                log(st, tcs)
            except (BrokenPipeError, ConnectionResetError):
                print("   client disconnected", flush=True)
            except ValueError as e:
                ev({"content": f"[error] {e}"}, "stop"); self.wfile.write(b"data: [DONE]\n\n")

    ThreadingHTTPServer((host, port), H).serve_forever()
