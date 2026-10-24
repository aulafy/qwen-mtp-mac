"""Banco del MoE de decode (batch 1) con las formas reales de DeepSeek-V4.1-Flash.

Pesos sintéticos en MXFP4 (bits aleatorios + escalas e8m0 realistas): para medir kernels
no hace falta el checkpoint, solo las formas y el formato. Compara:
  mlx-lm  : 3 gather_qmm (6 rutados) + experto compartido con 3 quantized_matmul (como mlx-lm)
  mlx-7   : los 7 expertos por gather_qmm (MLX con la misma fusión de compartido que el kernel)
  metal   : kernels.moe_decode (2 dispatches)
Uso: python bench_moe.py [--layers 8] [--reps 5]
"""
import argparse
import statistics
import time

import mlx.core as mx

from kernels import moe_decode

# DeepSeek-V4.1-Flash (config.json oficial, text_config)
H, I = 5120, 2304
TOPK, LIMIT, RSF = 6, 10.0, 1.5
GS = 32


def rand_mxfp4(shape_nk, key):
    *lead, n, k = shape_nk
    k1, k2 = mx.random.split(key)
    w = mx.random.randint(0, 2**31 - 1, (*lead, n, k // 8), key=k1).astype(mx.uint32)
    w = w | (mx.random.randint(0, 2, (*lead, n, k // 8), key=k2).astype(mx.uint32) << 31)
    s = mx.random.randint(118, 124, (*lead, n, k // GS), key=k2).astype(mx.uint8)
    return w, s


def make_layer(seed, n_alloc):
    key = mx.random.key(seed)
    kg, ku, kd, ki, ks = mx.random.split(key, 5)
    wg, sg = rand_mxfp4((n_alloc, I, H), kg)
    wu, su = rand_mxfp4((n_alloc, I, H), ku)
    wd, sd = rand_mxfp4((n_alloc, H, I), kd)
    perm = mx.random.permutation(n_alloc, key=ki)[: TOPK + 1].astype(mx.int32)
    sc = mx.softmax(mx.random.normal((TOPK,), key=ks))
    scores = mx.concatenate([sc / sc.sum() * RSF, mx.array([1.0])]).astype(mx.float32)
    L = dict(wg=wg, sg=sg, wu=wu, su=su, wd=wd, sd=sd, idx=perm, scores=scores)
    sh = int(perm[TOPK].item())
    # copia contigua del compartido para la ruta estilo mlx-lm (QuantizedLinear)
    L["shared"] = [mx.array(a[sh]) for a in (wg, sg, wu, su, wd, sd)]
    mx.eval(L)
    return L


def act(g, u):
    g = mx.minimum(g, LIMIT)
    u = mx.clip(u, -LIMIT, LIMIT)
    return g * mx.sigmoid(g) * u


def qmm(x, w, s):
    return mx.quantized_matmul(x, w, s, transpose=True, group_size=GS, bits=4, mode="mxfp4")


def gqmm(x, w, s, idx):
    return mx.gather_qmm(x, w, s, rhs_indices=idx, transpose=True, group_size=GS, bits=4, mode="mxfp4")


def moe_mlx_lm(x, L):
    idx = L["idx"][:TOPK].astype(mx.uint32)
    xx = x.reshape(1, 1, H)
    h = act(gqmm(xx, L["wg"], L["sg"], idx), gqmm(xx, L["wu"], L["su"], idx))
    y = gqmm(h, L["wd"], L["sd"], idx).reshape(TOPK, H)
    y = (y * L["scores"][:TOPK, None].astype(y.dtype)).sum(0)
    wg, sg, wu, su, wd, sd = L["shared"]
    xs = x.reshape(1, H)
    ys = qmm(act(qmm(xs, wg, sg), qmm(xs, wu, su)), wd, sd).reshape(H)
    return y + ys


def moe_mlx7(x, L):
    idx = L["idx"].astype(mx.uint32)
    xx = x.reshape(1, 1, H)
    h = act(gqmm(xx, L["wg"], L["sg"], idx), gqmm(xx, L["wu"], L["su"], idx))
    y = gqmm(h, L["wd"], L["sd"], idx).reshape(TOPK + 1, H)
    return (y * L["scores"][:, None].astype(y.dtype)).sum(0)


def moe_metal(x, L, **kw):
    return moe_decode(x, L["wg"], L["sg"], L["wu"], L["su"], L["wd"], L["sd"],
                      L["idx"], L["scores"], limit=LIMIT, **kw)


def run_chain(fn, x, layers, reps):
    # Como un decode real: un único eval por "token" que atraviesa todas las capas.
    y = x
    for _ in range(reps):
        for L in layers:
            y = y + (0.01 * fn(y, L)).astype(y.dtype)
    return y


def bench(fn, x, layers, reps, trials=7):
    mx.eval(run_chain(fn, x, layers, 1))  # calentar / compilar
    ts = []
    for _ in range(trials):
        t0 = time.perf_counter()
        mx.eval(run_chain(fn, x, layers, reps))
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) / (reps * len(layers))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    dt = getattr(mx, args.dtype)

    print(f"device: {mx.device_info()['device_name']}  mlx {mx.__version__}")
    layers = [make_layer(100 + i, TOPK + 2) for i in range(args.layers)]
    x = (mx.random.normal((H,), key=mx.random.key(0)) * 0.5).astype(dt)

    # exactitud contra MLX (misma entrada, misma capa)
    ref = moe_mlx_lm(x, layers[0]).astype(mx.float32)
    for name, fn in [("mlx-7", moe_mlx7), ("metal", moe_metal)]:
        out = fn(x, layers[0]).astype(mx.float32)
        err = mx.abs(out - ref).max().item() / (mx.abs(ref).max().item() + 1e-9)
        print(f"  {name:7s} err rel max vs mlx-lm: {err:.2e}")

    bytes_layer = (TOPK + 1) * 3 * I * H * (0.5 + 1 / GS)
    print(f"bytes de pesos por capa MoE: {bytes_layer/1e6:.1f} MB "
          f"(capas distintas: {args.layers}, evals por medida: {args.layers*args.reps})")
    res = {}
    for name, fn in [("mlx-lm", moe_mlx_lm), ("mlx-7", moe_mlx7),
                     ("mlx-lm+compile", mx.compile(moe_mlx_lm)), ("metal", moe_metal)]:
        t = bench(fn, x, layers, args.reps)
        res[name] = t
        print(f"  {name:15s} {t*1e3:7.3f} ms/capa  {bytes_layer/t/1e9:6.1f} GB/s")
    base = res["mlx-lm"]
    print(f"speedup metal vs mlx-lm: {base/res['metal']:.2f}x   vs mlx-lm+compile: "
          f"{res['mlx-lm+compile']/res['metal']:.2f}x")


if __name__ == "__main__":
    main()
