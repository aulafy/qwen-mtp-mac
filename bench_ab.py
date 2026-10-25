"""Banco A/B intercalado para Qwen3.5-9B: base (mlx-lm) vs MTP con k=1/k=2 borradores.

En un MacBook Air (sin ventilador) la GPU baja de frecuencia con carga sostenida, así que medir
configuraciones una detrás de otra sesga el resultado. Aquí se alternan prompt a prompt y ronda a
ronda, y se suman tokens y tiempos de cada configuración.
Uso: python bench_ab.py [--rounds 2] [--max-tokens 128] [--configs base,k2h,k2h8,k2ha] [--tau 0.15]
"""
import argparse
import time

import tuning  # noqa: F401  (antes que mlx)
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import stream_generate

import gdn_fused
import mtp as M
from lm_head2 import TwoStageHead
import qmv_kernel
import qwen_ollama

LM_HEAD = {"bits": 6, "group_size": 64}
MTP_Q = {"bits": 4, "group_size": 16, "mode": "nvfp4"}
DRAFT_HEAD = {"bits": 4, "group_size": 64}


def run_base(model, tok, ids, max_tokens):
    t0, n = None, 0
    for r in stream_generate(model, tok, ids, max_tokens=max_tokens):
        if t0 is None:  # excluye el prefill, como generate_mtp
            t0 = time.perf_counter()
        n += 1
    return n - 1, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--configs", default="base,k2h,k2h8,k2ha",
                    help="base o kN + letras: h lm_head en dos fases, 8 MTP a 8 bits, a k adaptativo, "
                         "f borrador con vocabulario completo, g GatedDeltaNet fusionada")
    ap.add_argument("--tau", type=float, default=0.15, help="umbral del k adaptativo")
    ap.add_argument("--draft-vocab", type=int, default=65536, help="ids frecuentes del lm_head de borrador")
    args = ap.parse_args()

    qmv_kernel.install()
    model, tok = qwen_ollama.load()  # lm_head bf16: de él salen la copia de 6 bits y el de dos fases
    lm = model.language_model
    lm_heads = {"6": nn.QuantizedLinear.from_linear(lm.lm_head, **LM_HEAD), "h": TwoStageHead(lm.lm_head.weight)}
    mx.eval([h.parameters() for h in lm_heads.values()])
    lm.lm_head = lm_heads["6"]
    mtps = {"4": M.load_mtp(lm, MTP_Q), "8": M.load_mtp(lm, {"bits": 8, "group_size": 64})}
    eos = set(tok.eos_token_ids)
    # kN = MTP con N borradores y lm_head de borrador reducido (ids frecuentes); kNf = vocabulario completo;
    # kNg = además la GatedDeltaNet fusionada (gdn_fused.py); kNh = lm_head en dos fases (lm_head2.py)
    heads = {"": M.load_draft_head(args.draft_vocab, DRAFT_HEAD, freq=True, extra_ids=eos),
             "f": M.load_draft_head(248320, DRAFT_HEAD)}
    print(f"mem {mx.get_active_memory()/1e9:.2f} GB")

    configs = args.configs.split(",")
    tot = {c: [0, 0.0] for c in configs}
    cyc = {c: [0.0, 0] for c in configs}
    bc = {c: [0.0, 0] for c in configs}  # borradores verificados por ciclo
    for rnd in range(args.rounds):
        for i, prompt in enumerate(M.PROMPTS):
            ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          add_generation_prompt=True, enable_thinking=False)
            order = configs if (rnd + i) % 2 == 0 else configs[::-1]
            line = []
            for c in order:
                lm.lm_head = lm_heads["h" if "h" in c[2:] else "6"]
                if c == "base":
                    n, dt = run_base(model, tok, ids, args.max_tokens)
                else:  # kN[f][g][h][8][a]: ver la ayuda de --configs
                    flags = c[2:]
                    (gdn_fused.install if "g" in flags else gdn_fused.uninstall)()
                    head = heads["f" if "f" in flags else ""]
                    out, st = M.generate_mtp(lm, mtps["8" if "8" in flags else "4"], ids, args.max_tokens, eos,
                                             head, k=int(c[1]), tau=args.tau if "a" in flags else None)
                    bc[c][0] += st["borradores_ciclo"]
                    bc[c][1] += 1
                    n, dt = len(out) - 1, (len(out) - 1) / st["tps"]
                    cyc[c][0] += st["tok_ciclo"]
                    cyc[c][1] += 1
                tot[c][0] += n
                tot[c][1] += dt
                line.append(f"{c} {n/dt:5.1f}")
            print(f"  ronda {rnd} prompt {i}: " + "  ".join(line), flush=True)
    base = tot["base"][0] / tot["base"][1] if "base" in tot else None
    for c in configs:
        tps = tot[c][0] / tot[c][1]
        extra = f"  {cyc[c][0]/cyc[c][1]:.2f} tok/ciclo  {bc[c][0]/bc[c][1]:.2f} borr/ciclo" if cyc[c][1] else ""
        rel = f"  ({tps/base:.2f}x)" if base else ""
        print(f"{c:5s} {tps:5.1f} tok/s{rel}{extra}")


if __name__ == "__main__":
    main()
