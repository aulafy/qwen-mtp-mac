"""Prueba del servidor con los clientes oficiales de OpenAI y Ollama (python server.py antes).

  python test_server.py [--port 11435]
"""
import argparse
import time

import ollama
from openai import OpenAI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11435)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    oa = OpenAI(base_url=base + "/v1", api_key="local")
    ol = ollama.Client(host=base)

    print("modelos:", [m.id for m in oa.models.list().data], [m.model for m in ol.list().models])

    # 1) OpenAI, no stream, greedy
    r = oa.chat.completions.create(model="qwen3.5-mtp:9b", temperature=0, max_tokens=60,
                                   messages=[{"role": "user", "content": "Di solo: hola mundo"}])
    print("openai greedy:", repr(r.choices[0].message.content), r.choices[0].finish_reason, r.usage.completion_tokens)

    # 2) OpenAI, stream, chat de 2 turnos: el segundo debe reutilizar la caché del primero
    msgs = [{"role": "system", "content": "Eres un asistente conciso."},
            {"role": "user", "content": "Explica en 3 frases qué es la fotosíntesis."}]
    for turno in range(2):
        t0 = time.perf_counter()
        text, usage = "", None
        for ch in oa.chat.completions.create(model="m", messages=msgs, stream=True, temperature=0,
                                             max_tokens=200, stream_options={"include_usage": True}):
            if ch.choices and ch.choices[0].delta.content:
                text += ch.choices[0].delta.content
            if ch.usage:
                usage = ch.usage
        dt = time.perf_counter() - t0
        cached = usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else 0
        print(f"turno {turno}: {usage.completion_tokens} tok en {dt:.1f}s, prompt {usage.prompt_tokens} "
              f"(caché {cached}) -> {text[:80]!r}")
        msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": "Ahora en una sola frase."}]

    # 3) stop string
    r = oa.chat.completions.create(model="m", temperature=0, max_tokens=100, stop=["3"],
                                   messages=[{"role": "user", "content": "Cuenta del 1 al 10 separados por comas."}])
    print("stop '3':", repr(r.choices[0].message.content), r.choices[0].finish_reason)

    # 4) Ollama chat con thinking
    r = ol.chat(model="qwen3.5-mtp:9b", think=True, options={"temperature": 0, "num_predict": 400},
                messages=[{"role": "user", "content": "¿Cuánto es 17*23? Responde solo el número."}])
    print("ollama think: thinking", len(r.message.thinking or ""), "car. | content:", repr(r.message.content))

    # 5) Ollama generate en streaming
    out = "".join(p.response for p in ol.generate(model="m", prompt="Completa: El cielo es", stream=True,
                                                     options={"temperature": 0, "num_predict": 20}))
    print("ollama generate:", repr(out))

    # 6) velocidad: 256 tokens con temperature 0 y con los valores por defecto (T=1, top_k 20, top_p 0.95)
    for temp in (0, None):
        opts = {"num_predict": 256} | ({"temperature": temp} if temp is not None else {})
        r = ol.chat(model="m", options=opts, messages=[{"role": "user", "content":
                    "Explica en detalle cómo funciona un motor de combustión interna de cuatro tiempos."}])
        print(f"velocidad T={'por defecto (1)' if temp is None else temp}: {r.eval_count} tok, "
              f"{r.eval_count / (r.eval_duration / 1e9):.1f} tok/s")


if __name__ == "__main__":
    main()
