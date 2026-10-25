"""Lab: coste de las 24 GatedDeltaNet de Qwen3.5-9B en decode/verificación (S = 1..4 tokens).

Separa lo que es leer pesos (proyecciones) del tramo central (conv, normas, compuertas, delta rule,
norma con compuerta), que son muchas operaciones pequeñas. Uso: python lab_gdn.py [--fused]
"""
import argparse
import statistics
import time

import tuning  # noqa: F401  (antes que mlx)
import mlx.core as mx

import mtp as M
import qmv_kernel
import qwen_ollama


def timeit(fn, n=15):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", action="store_true", help="usar el kernel fusionado (gdn_fused.py)")
    args = ap.parse_args()
    qmv_kernel.install()
    if args.fused:
        import gdn_fused
        gdn_fused.install()
    model, tok = qwen_ollama.load(lm_head_quant={"bits": 6, "group_size": 64})
    lm = model.language_model
    cache = lm.make_cache()
    p = mx.array(tok.encode("Prueba de rendimiento de las capas GatedDeltaNet en el Mac. " * 8))[None]
    lg, _, _ = M.backbone(lm, p, cache)
    mx.eval(lg)
    pairs = [(l.linear_attn, c) for l, c in zip(lm.model.layers, cache) if l.is_linear]
    kv = [c for l, c in zip(lm.model.layers, cache) if not l.is_linear]
    la0 = pairs[0][0]
    print("tipos:", {n: type(getattr(la0, n)).__name__ for n in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")})

    def restore(saved):
        for (_, c), (a, b) in zip(pairs, saved):
            c[0], c[1] = a, b

    for S in (1, 2, 3, 4):
        x = (mx.random.normal((1, S, 4096)) * 0.5).astype(mx.bfloat16)
        xv = (mx.random.normal((1, S, 4096)) * 0.5).astype(mx.bfloat16)
        saved = [(c[0], c[1]) for _, c in pairs]

        def full():
            for _, c in pairs:
                c.snap = S > 1
            outs = [la(x, None, c) for la, c in pairs]
            mx.eval(outs)
            for _, c in pairs:
                c.snap = False
            restore(saved)

        def proj():
            outs = []
            for la, _ in pairs:
                outs += [la.in_proj_qkv(x), la.in_proj_z(x), la.in_proj_a(x), la.in_proj_b(x), la.out_proj(xv)]
            mx.eval(outs)

        def backbone():
            for _, c in pairs:
                c.snap = S > 1
            lg, _, _ = M.backbone(lm, mx.array([list(range(13, 13 + S))]), cache)
            mx.eval(lg)
            for _, c in pairs:
                c.snap = False
            for c in kv:
                c.trim(S)
            restore(saved)

        tf, tp, tb = timeit(full), timeit(proj), timeit(backbone)
        print(f"S={S}: 24 GDN {tf:5.1f} ms = proyecciones {tp:5.1f} + tramo central {tf - tp:5.1f} "
              f"| pasada completa {tb:5.1f} ms", flush=True)


if __name__ == "__main__":
    main()
