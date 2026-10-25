"""Servidor local compatible con las APIs de Ollama y OpenAI para Qwen3.5-9B acelerado (engine.py).

  python server.py [--port 11435] [--k 2] [--think]

Chat:    GET /chat (interfaz web: ui/chat.html, datos en datos/)
Ollama:  GET /  /api/version  /api/tags   POST /api/chat  /api/generate  /api/show
OpenAI:  GET /v1/models   POST /v1/chat/completions  /v1/completions   (stream o no)
num_ctx (options.num_ctx en Ollama, num_ctx en OpenAI) recorta los mensajes más antiguos.

Una generación a la vez (la GPU es una); las peticiones concurrentes esperan turno. Solo escucha en
127.0.0.1. Sin tools/function calling, sin imágenes y sin presence/repeat penalty.
"""
import argparse
import base64
import copy
import json
import os
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import ui_api
from engine import Sampling, Worker

MODEL = "qwen3.5-mtp:9b"
DEFAULTS = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0}  # params de Ollama para qwen3.5
MAX_TOKENS = 4096
THINK_END = "</think>"
ENGINE = None  # engine.Engine (vive en el hilo de WORKER)
WORKER: Worker = None
ARGS = None
UI_HTML = Path(__file__).parent / "ui" / "chat.html"
ACTIVITY = {"last": time.time(), "active": 0}  # para --idle-exit


class ContextError(ValueError):
    pass


def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _partial(buf, markers):
    """Longitud del sufijo más largo de buf que es prefijo propio de algún marcador."""
    best = 0
    for m in markers:
        for n in range(min(len(m) - 1, len(buf)), best, -1):
            if buf.endswith(m[:n]):
                best = n
                break
    return best


class TextOut:
    """Detokeniza en streaming, separa <think>...</think> y corta en las stop strings."""

    def __init__(self, tok, think, stops):
        self.detok = copy.copy(tok.detokenizer)  # el del tokenizer es compartido
        self.detok.reset()
        self.in_think = think
        self.stops = [s for s in (stops or []) if s]
        self.buf = ""
        self.stopped = False
        self.lead = True  # quitar los "\n" que siguen a </think> (o al inicio de la respuesta)

    def push(self, token):
        self.detok.add_token(token)
        return self._feed(self.detok.last_segment, final=False)

    def finish(self):
        self.detok.finalize()
        return self._feed(self.detok.last_segment, final=True)

    def _feed(self, seg, final):
        self.buf += seg
        out = []
        while self.buf and not self.stopped:
            if self.in_think:
                i = self.buf.find(THINK_END)
                if i >= 0:
                    if i:
                        out.append(("thinking", self.buf[:i]))
                    self.buf = self.buf[i + len(THINK_END):]
                    self.in_think = False
                    continue
                cut = len(self.buf) - (0 if final else _partial(self.buf, [THINK_END]))
                if cut > 0:
                    out.append(("thinking", self.buf[:cut]))
                    self.buf = self.buf[cut:]
                break
            if self.lead:
                self.buf = self.buf.lstrip("\n")
                if not self.buf:
                    break
                self.lead = False
            hits = [(self.buf.find(s), s) for s in self.stops if s in self.buf]
            if hits:
                i = min(hits)[0]
                if i:
                    out.append(("content", self.buf[:i]))
                self.buf = ""
                self.stopped = True
                break
            cut = len(self.buf) - (0 if final else _partial(self.buf, self.stops))
            if cut > 0:
                out.append(("content", self.buf[:cut]))
                self.buf = self.buf[cut:]
            break
        return out


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # partes estilo OpenAI
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return "" if content is None else str(content)


CTX_MSG = {
    "es": "El mensaje no cabe en el contexto: {n} tokens + {r} de respuesta > {c}. Sube el contexto o quita archivos.",
    "en": "The message does not fit in the context: {n} tokens + {r} for the answer > {c}. "
          "Raise the context or remove files.",
}


def build_chat(messages, think, num_ctx=0, reserve=0, lang="en"):
    """Plantilla de chat -> (ids, prefijo reutilizable, mensajes antiguos omitidos).
    Con num_ctx, quita los mensajes más antiguos (no los system ni el último) hasta que el prompt
    más `reserve` tokens de respuesta quepan."""
    msgs = [{"role": m.get("role", "user"), "content": text_of(m.get("content"))} for m in messages]
    tok = ENGINE.tok
    dropped = 0
    while True:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=think)
        if not num_ctx or len(ids) + reserve <= num_ctx:
            break
        drop = next((i for i, m in enumerate(msgs[:-1]) if m["role"] != "system"), None)
        if drop is None:
            raise ContextError(CTX_MSG.get(lang, CTX_MSG["en"]).format(n=len(ids), r=reserve, c=num_ctx))
        del msgs[drop]
        dropped += 1
    nogen = tok.apply_chat_template(msgs, add_generation_prompt=False, enable_thinking=think)
    n = 0
    while n < min(len(ids), len(nogen)) and ids[n] == nogen[n]:
        n += 1
    return ids, n, dropped


def sampling_from(opts):
    v = {**DEFAULTS, **{k: opts[k] for k in DEFAULTS if opts.get(k) is not None}}
    if ARGS.greedy and opts.get("temperature") is None:
        v["temperature"] = 0.0
    return Sampling(temperature=float(v["temperature"]), top_k=int(v["top_k"]), top_p=float(v["top_p"]),
                    min_p=float(v["min_p"]), seed=opts.get("seed"))


def run(ids, prefix_len, max_tokens, sp, think, stops, on_piece):
    """Genera y llama on_piece(tipo, texto). Devuelve (finish_reason, métricas)."""
    finish, n = "length", 0
    out = TextOut(ENGINE.tok, think, stops)
    t0 = time.perf_counter()
    job = WORKER.submit(ids, prefix_len, max_tokens, sp)
    ACTIVITY["active"] += 1
    try:
        for tid in job.tokens():
            n += 1
            if tid in ENGINE.eos:
                finish = "stop"
                break
            for kind, text in out.push(tid):
                on_piece(kind, text)
            if out.stopped:
                finish = "stop"
                break
        for kind, text in out.finish():
            on_piece(kind, text)
    finally:
        job.cancel.set()
        for _ in job.tokens():  # espera a que el worker suelte el modelo
            pass
        ACTIVITY["active"] -= 1
        ACTIVITY["last"] = time.time()
    t1 = time.perf_counter()
    stats = job.stats
    t_start, t_first = stats.get("t_start", t0), stats.get("t_first", t0)
    stats.update(completion_tokens=n, queue_s=t_start - t0, prompt_s=t_first - t_start,
                 gen_s=max(t1 - t_first, 1e-9), total_s=t1 - t0)
    gen_s = stats["gen_s"]
    wait = f" | esperó {stats['queue_s']:.1f}s" if stats["queue_s"] > 0.05 else ""
    print(f"[{MODEL}] prompt {len(ids)} tok (caché {stats.get('cached_tokens', 0)}) en "
          f"{stats['prompt_s']:.2f}s | {n} tok a {n / gen_s:.1f} tok/s | T={sp.temperature} | {finish}{wait}",
          flush=True)
    return finish, stats


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "qwen-mtp-mac"

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *a):
        if ARGS.verbose:
            super().log_message(fmt, *a)

    # ---- utilidades HTTP ----
    def _origin_ok(self):
        """Como Ollama: solo orígenes locales (o sin Origin, p. ej. curl y SDKs). Evita que una web
        cualquiera use el modelo o lea tus chats; el Host evita el DNS rebinding."""
        host = urlparse("//" + (self.headers.get("Host") or "")).hostname or ""
        if ARGS.host in ("127.0.0.1", "localhost") and host not in ("127.0.0.1", "localhost", "::1"):
            return False
        origin = self.headers.get("Origin")
        if not origin or origin == "null":
            return True
        return (urlparse(origin).hostname or "") in ("127.0.0.1", "localhost", "::1")

    def _lang(self, body=None):
        """Idioma de los mensajes de error para la interfaz (es/en); inglés para el resto de clientes."""
        lang = (body or {}).get("lang") or self.headers.get("X-Lang") or "en"
        return lang if lang in ("es", "en") else "en"

    def _cors(self):
        origin = self.headers.get("Origin")
        if origin and origin != "null":
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _send(self, code, body, ctype="application/json"):
        if not isinstance(body, bytes):
            body = (json.dumps(body, ensure_ascii=False) if ctype == "application/json" else body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _stream_start(self, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self._cors()
        self.end_headers()

    def _chunk(self, data):
        data = data.encode() if isinstance(data, str) else data
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _stream_end(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # ---- rutas ----
    def do_OPTIONS(self):
        if not self._origin_ok():
            return self._send(403, {"error": "origin not allowed"})
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Lang")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._origin_ok():
            return self._send(403, {"error": "origin not allowed"})
        ACTIVITY["last"] = time.time()
        if self.path == "/":
            return self._send(200, "Ollama is running", "text/plain; charset=utf-8")
        if self.path in ("/chat", "/chat/", "/ui"):
            return self._send(200, UI_HTML.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/version":
            return self._send(200, {"version": "0.30.10-qwen-mtp-mac"})
        if self.path == "/api/tags":
            return self._send(200, {"models": [self._model_card()]})
        if self.path == "/api/ps":
            return self._send(200, {"models": [{**self._model_card(), "expires_at": "2099-01-01T00:00:00Z"}]})
        if self.path == "/v1/models":
            return self._send(200, {"object": "list", "data": [
                {"id": MODEL, "object": "model", "created": int(time.time()), "owned_by": "qwen-mtp-mac"}]})
        if self.path == "/ui/api/chats":
            return self._send(200, ui_api.list_chats())
        if self.path.startswith("/ui/api/chats/"):
            chat = self._ui(lambda: ui_api.get_chat(self.path.rsplit("/", 1)[1]))
            if chat is not False:
                self._send(200, chat) if chat else self._send(404, {"error": "not found"})
            return
        if self.path == "/ui/api/memory":
            return self._send(200, {"text": ui_api.get_memory()})
        if self.path == "/ui/api/info":
            return self._send(200, {"model": MODEL, "k": ARGS.k, "idle_exit_min": ARGS.idle_exit,
                                    "defaults": DEFAULTS})
        self._send(404, {"error": f"not found: {self.path}"})

    def _ui(self, fn):
        try:
            return fn()
        except ValueError as e:
            self._send(400, {"error": str(e)})
            return False

    def do_PUT(self):
        if not self._origin_ok():
            return self._send(403, {"error": "origin not allowed"})
        body = self._body()
        if self.path.startswith("/ui/api/chats/"):
            chat = self._ui(lambda: ui_api.save_chat(self.path.rsplit("/", 1)[1], body))
            if chat is not False:
                self._send(200, {"ok": True, "updated": chat["updated"]})
            return
        if self.path == "/ui/api/memory":
            ui_api.set_memory(body.get("text", ""))
            return self._send(200, {"ok": True})
        self._send(404, {"error": f"not found: {self.path}"})

    def do_DELETE(self):
        if not self._origin_ok():
            return self._send(403, {"error": "origin not allowed"})
        if self.path.startswith("/ui/api/chats/"):
            if self._ui(lambda: ui_api.delete_chat(self.path.rsplit("/", 1)[1])) is not False:
                self._send(200, {"ok": True})
            return
        self._send(404, {"error": f"not found: {self.path}"})

    def do_POST(self):
        if not self._origin_ok():
            return self._send(403, {"error": "origin not allowed"})
        ACTIVITY["last"] = time.time()
        try:
            body = self._body()
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON"})
        routes = {"/api/chat": self.ollama_chat, "/api/generate": self.ollama_generate,
                  "/api/show": self.ollama_show, "/v1/chat/completions": self.openai_chat,
                  "/v1/completions": self.openai_completions, "/ui/api/upload": self.ui_upload,
                  "/ui/api/shutdown": self.ui_shutdown}
        fn = routes.get(self.path)
        if fn is None:
            return self._send(404, {"error": f"not found: {self.path}"})
        try:
            fn(body)
        except ContextError as e:
            self._send(400, {"error": str(e)})
        except (BrokenPipeError, ConnectionResetError):
            print("cliente desconectado: generación cortada", flush=True)

    def ui_upload(self, body):
        try:
            data = base64.b64decode(body.get("data") or "")
            text, truncated = ui_api.extract_text(body.get("name") or "file", data, self._lang(body))
        except ValueError as e:
            return self._send(400, {"error": f"{body.get('name')}: {e}"})
        self._send(200, {"name": body.get("name"), "text": text, "truncated": truncated,
                         "tokens": len(ENGINE.tok.encode(text))})

    def ui_shutdown(self, body):
        self._send(200, {"ok": True})
        print("apagado desde la interfaz", flush=True)
        threading.Timer(0.3, os._exit, (0,)).start()

    def _model_card(self):
        return {"name": MODEL, "model": MODEL, "modified_at": now_iso(), "size": 7_000_000_000,
                "digest": "qwen-mtp-mac", "details": {"format": "safetensors", "family": "qwen35",
                                                  "families": ["qwen35"], "parameter_size": "9B",
                                                  "quantization_level": "NVFP4"}}

    def ollama_show(self, body):
        self._send(200, {"modelfile": "", "parameters": "", "template": "", "details": self._model_card()["details"],
                         "model_info": {"general.architecture": "qwen35"}, "capabilities": ["completion", "thinking"]})

    # ---- Ollama ----
    def _ollama(self, body, ids, prefix_len, key, dropped=0):
        opts = body.get("options") or {}
        think = bool(body.get("think", ARGS.think))
        sp = sampling_from(opts)
        npred = opts.get("num_predict")
        max_tokens = MAX_TOKENS if npred in (None, -1, -2) else int(npred)
        stops = opts.get("stop") or []
        stream = body.get("stream", True)

        def msg(kind, text):
            if key == "message":
                m = {"role": "assistant", "content": text if kind == "content" else ""}
                if kind == "thinking":
                    m["thinking"] = text
                return {"message": m}
            return {"response": text if kind == "content" else "", **({"thinking": text} if kind == "thinking" else {})}

        parts = {"content": [], "thinking": []}
        if stream:
            self._stream_start("application/x-ndjson")

        def on_piece(kind, text):
            if stream:
                self._chunk(json.dumps({"model": MODEL, "created_at": now_iso(), **msg(kind, text),
                                        "done": False}, ensure_ascii=False) + "\n")
            else:
                parts[kind].append(text)

        finish, st = run(ids, prefix_len, max_tokens, sp, think, stops, on_piece)
        ns = lambda s: int(s * 1e9)
        final = {"model": MODEL, "created_at": now_iso(), "done": True, "done_reason": finish,
                 "total_duration": ns(st["total_s"]), "load_duration": 0,
                 "prompt_eval_count": st["prompt_tokens"], "prompt_eval_duration": ns(st["prompt_s"]),
                 "eval_count": st["completion_tokens"], "eval_duration": ns(st["gen_s"]),
                 "cached_tokens": st.get("cached_tokens", 0), "context_dropped": dropped}
        if stream:
            final.update(msg("content", ""))
            self._chunk(json.dumps(final, ensure_ascii=False) + "\n")
            self._stream_end()
        else:
            full = msg("content", "".join(parts["content"]))
            if parts["thinking"]:
                (full["message"] if key == "message" else full)["thinking"] = "".join(parts["thinking"])
            self._send(200, {**final, **full})

    def ollama_chat(self, body):
        opts = body.get("options") or {}
        npred = opts.get("num_predict")
        reserve = 0 if npred in (None, -1, -2) else int(npred)
        ids, n, dropped = build_chat(body.get("messages") or [], bool(body.get("think", ARGS.think)),
                                     int(opts.get("num_ctx") or 0), reserve, self._lang(body))
        self._ollama(body, ids, n, "message", dropped)

    def ollama_generate(self, body):
        prompt = body.get("prompt") or ""
        if body.get("raw"):
            ids = ENGINE.tok.encode(prompt)
            n = len(ids) - 1
        else:
            msgs = ([{"role": "system", "content": body["system"]}] if body.get("system") else []) + \
                   [{"role": "user", "content": prompt}]
            ids, n, _ = build_chat(msgs, bool(body.get("think", ARGS.think)))
        self._ollama(body, ids, n, "response")

    # ---- OpenAI ----
    def _openai(self, body, ids, prefix_len, chat):
        think = bool((body.get("chat_template_kwargs") or {}).get("enable_thinking", body.get("think", ARGS.think)))
        sp = sampling_from(body)
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or MAX_TOKENS)
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else stops
        stream = bool(body.get("stream", False))
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"
        parts = {"content": [], "thinking": []}

        def chunk(delta=None, text=None, finish=None):
            c = {"index": 0, "finish_reason": finish}
            if chat:
                c["delta"] = delta or {}
            else:
                c["text"] = text or ""
            return {"id": rid, "object": obj + (".chunk" if chat else ""), "created": created,
                    "model": MODEL, "choices": [c]}

        if stream:
            self._stream_start("text/event-stream")
            if chat:
                self._chunk("data: " + json.dumps(chunk({"role": "assistant", "content": ""})) + "\n\n")

        def on_piece(kind, text):
            if not stream:
                parts[kind].append(text)
                return
            if chat:
                d = {"content": text} if kind == "content" else {"reasoning_content": text}
                self._chunk("data: " + json.dumps(chunk(d), ensure_ascii=False) + "\n\n")
            elif kind == "content":
                self._chunk("data: " + json.dumps(chunk(text=text), ensure_ascii=False) + "\n\n")

        finish, st = run(ids, prefix_len, max_tokens, sp, think, stops, on_piece)
        usage = {"prompt_tokens": st["prompt_tokens"], "completion_tokens": st["completion_tokens"],
                 "total_tokens": st["prompt_tokens"] + st["completion_tokens"],
                 "prompt_tokens_details": {"cached_tokens": st.get("cached_tokens", 0)}}
        if stream:
            self._chunk("data: " + json.dumps(chunk(finish=finish)) + "\n\n")
            if (body.get("stream_options") or {}).get("include_usage"):
                u = chunk()
                u["choices"], u["usage"] = [], usage
                self._chunk("data: " + json.dumps(u) + "\n\n")
            self._chunk("data: [DONE]\n\n")
            self._stream_end()
            return
        content = "".join(parts["content"])
        if chat:
            m = {"role": "assistant", "content": content}
            if parts["thinking"]:
                m["reasoning_content"] = "".join(parts["thinking"])
            choice = {"index": 0, "message": m, "finish_reason": finish}
        else:
            choice = {"index": 0, "text": content, "finish_reason": finish}
        self._send(200, {"id": rid, "object": obj, "created": created, "model": MODEL,
                         "choices": [choice], "usage": usage})

    def openai_chat(self, body):
        think = bool((body.get("chat_template_kwargs") or {}).get("enable_thinking", body.get("think", ARGS.think)))
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 0)
        ids, n, _ = build_chat(body.get("messages") or [], think, int(body.get("num_ctx") or 0), max_tokens,
                               self._lang(body))
        self._openai(body, ids, n, chat=True)

    def openai_completions(self, body):
        prompt = body.get("prompt") or ""
        prompt = prompt[0] if isinstance(prompt, list) else prompt
        ids = ENGINE.tok.encode(prompt)
        self._openai(body, ids, len(ids) - 1, chat=False)


def main():
    global ENGINE, WORKER, ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("QWEN_MTP_PORT", 11435)))
    ap.add_argument("--k", type=int, default=2, help="borradores MTP por ciclo")
    ap.add_argument("--lm-head", default="2f", help="2f (dos fases, por defecto) o bits: 6, 8...")
    ap.add_argument("--think", action="store_true", help="razonamiento activado por defecto")
    ap.add_argument("--greedy", action="store_true", help="temperature 0 si la petición no la indica")
    ap.add_argument("--idle-exit", type=float, default=0,
                    help="minutos sin uso tras los que el servidor se cierra y libera la RAM (0 = nunca)")
    ap.add_argument("--verbose", action="store_true")
    ARGS = ap.parse_args()
    t0 = time.time()
    WORKER = Worker(k=ARGS.k, lm_head=ARGS.lm_head)
    ENGINE = WORKER.engine
    print(f"{MODEL} cargado en {time.time() - t0:.1f}s; escuchando en http://{ARGS.host}:{ARGS.port} "
          f"(chat: /chat, Ollama: /api/*, OpenAI: /v1/*)", flush=True)
    if ARGS.idle_exit > 0:
        threading.Thread(target=_idle_watch, daemon=True).start()
    ThreadingHTTPServer((ARGS.host, ARGS.port), Handler).serve_forever()


def _idle_watch():
    while True:
        time.sleep(20)
        if ACTIVITY["active"] == 0 and time.time() - ACTIVITY["last"] > ARGS.idle_exit * 60:
            print(f"{ARGS.idle_exit:g} min sin uso: me cierro para liberar la memoria", flush=True)
            os._exit(0)


if __name__ == "__main__":
    main()
