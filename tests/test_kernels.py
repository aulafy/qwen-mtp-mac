"""Tests de exactitud de los kernels (no necesitan el modelo ni Ollama; sí un Mac con Apple Silicon).

    python -m pytest -q tests
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tuning  # noqa: F401,E402  (antes que mlx)
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_lm.models import qwen3_5  # noqa: E402
from mlx_lm.models.cache import ArraysCache  # noqa: E402
from mlx_lm.models.gated_delta import compute_g, gated_delta_update  # noqa: E402

import gdn_fused  # noqa: E402
import mtp as M  # noqa: E402
from engine import Engine, Sampling  # noqa: E402
from gdn_kernel import gated_delta_snapshot  # noqa: E402
from lm_head2 import TwoStageHead  # noqa: E402
from qmv_kernel import qmv_nvfp4  # noqa: E402


@pytest.mark.parametrize("M_rows", [1, 2, 3, 4])
def test_qmv_nvfp4_matches_mlx(M_rows):
    N, K = 1024, 2048
    w, s = mx.quantize(mx.random.normal((N, K), key=mx.random.key(0)) / K**0.5, group_size=16, bits=4, mode="nvfp4")
    x = mx.random.normal((1, M_rows, K), key=mx.random.key(1)).astype(mx.bfloat16)
    ref = mx.quantized_matmul(x, w, s, transpose=True, group_size=16, bits=4, mode="nvfp4").astype(mx.float32)
    out = qmv_nvfp4(x, w, s).astype(mx.float32)
    assert out.shape == ref.shape
    # mismo resultado salvo el orden de las sumas (redondeo final a bf16)
    assert (mx.abs(out - ref).max() / mx.abs(ref).max()).item() < 1e-2


def test_gdn_snapshot_matches_two_mlx_calls():
    B, T, Hk, Hv, D = 1, 4, 16, 32, 128
    ks = mx.random.split(mx.random.key(3), 7)
    q = mx.random.normal((B, T, Hk, D), key=ks[0]).astype(mx.bfloat16) * 0.1
    k = mx.random.normal((B, T, Hk, D), key=ks[1]).astype(mx.bfloat16) * 0.1
    v = mx.random.normal((B, T, Hv, D), key=ks[2]).astype(mx.bfloat16)
    a = mx.random.normal((B, T, Hv), key=ks[3]).astype(mx.bfloat16)
    b = mx.random.normal((B, T, Hv), key=ks[4]).astype(mx.bfloat16)
    A_log = mx.random.normal((Hv,), key=ks[5])
    dt = mx.random.normal((Hv,), key=ks[6]).astype(mx.bfloat16)
    st = mx.random.normal((B, Hv, D, D)) * 0.1
    y, s_final, mids = gated_delta_snapshot(q, k, v, compute_g(A_log, a, dt), mx.sigmoid(b), st)
    s, ys = st, []
    for t in range(T):
        o, s = gated_delta_update(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1], a[:, t:t + 1], b[:, t:t + 1],
                                  A_log, dt, s)
        ys.append(o)
        if t < T - 1:
            assert mx.array_equal(mids[t], s).item()
    assert mx.array_equal(y, mx.concatenate(ys, axis=1)).item()
    assert mx.array_equal(s_final, s).item()


def _gdn_layer(seed=0):
    args = qwen3_5.TextModelArgs(hidden_size=512, linear_num_value_heads=8, linear_num_key_heads=4,
                                 linear_key_head_dim=128, linear_value_head_dim=128, rms_norm_eps=1e-6)
    mx.random.seed(seed)
    layer = qwen3_5.GatedDeltaNet(args)
    layer.set_dtype(mx.bfloat16)
    layer.A_log = mx.log(mx.random.uniform(1, 16, (args.linear_num_value_heads,)))  # f32 como en el modelo
    layer.dt_bias = mx.random.normal((args.linear_num_value_heads,)).astype(mx.bfloat16)
    layer.norm.weight = (1 + 0.2 * mx.random.normal((128,))).astype(mx.bfloat16)
    layer.eval()  # en modo entrenamiento mlx-lm usa su ruta de referencia en vez del kernel Metal
    return layer


@pytest.mark.parametrize("S", [1, 3])
def test_gdn_fused_is_bit_exact(S):
    layer = _gdn_layer()
    conv0 = mx.random.normal((1, layer.conv_kernel_size - 1, layer.conv_dim)).astype(mx.bfloat16)
    ssm0 = mx.random.normal((1, layer.num_v_heads, layer.head_v_dim, layer.head_k_dim)) * 0.1
    x = mx.random.normal((1, S, 512)).astype(mx.bfloat16)
    res = {}
    for name, call in (("mlx", M._gdn_call), ("fused", gdn_fused._call)):
        c = ArraysCache(size=2)
        c[0], c[1] = conv0, ssm0
        c.snap = S > 1
        out = call(layer, x, None, c)
        mx.eval(out, c[0], c[1])
        res[name] = (out, c[0], c[1], getattr(c, "rollback", None))
    (o1, c1, s1, r1), (o2, c2, s2, r2) = res["mlx"], res["fused"]
    assert mx.array_equal(o1, o2).item()
    assert mx.array_equal(c1, c2).item()
    assert mx.array_equal(s1, s2).item()
    if S > 1:
        assert mx.array_equal(r1[1], r2[1]).item()  # instantáneas para deshacer


def test_bf16_formulas_match_mlx_exhaustively():
    """sigmoid, silu y softplus de gdn_fused (redondeo bf16 tras cada op) = las de MLX en los 65.536 bf16."""
    x = mx.array(np.arange(65536, dtype=np.uint16)).view(mx.bfloat16)
    src = r"""
      uint i = thread_position_in_grid.x; float v = float(x[i]);
      sg[i] = sigmoid_bf(v); sl[i] = silu_bf(v); sp[i] = softplus_bf(v);
    """
    k = mx.fast.metal_kernel(name="probe_bf16", input_names=["x"], output_names=["sg", "sl", "sp"],
                             source=src, header=gdn_fused.BF16_MATH)
    sg, sl, sp = k(inputs=[x], grid=(65536, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(65536,)] * 3,
                   output_dtypes=[mx.float32] * 3)
    finite = mx.isfinite(x)
    for got, ref in ((sg, mx.sigmoid(x)), (sl, nn.silu(x)), (sp, nn.softplus(x))):
        ref = ref.astype(mx.float32)
        same = (got == ref) | (mx.isnan(got) & mx.isnan(ref))
        assert mx.where(finite, same, True).all().item()


def test_two_stage_head_matches_full_bf16():
    V, D = 16384, 512
    W = (mx.random.normal((V, D), key=mx.random.key(5)) / D**0.5).astype(mx.bfloat16)
    h = mx.random.normal((1, 64, D), key=mx.random.key(6)).astype(mx.bfloat16)
    head = TwoStageHead(W, k=64)
    full = (h @ W.T).astype(mx.float32)
    two = head(h).astype(mx.float32)
    cand = mx.isfinite(two)
    assert cand.sum(-1).min().item() == 64
    assert mx.array_equal(mx.where(cand, two, 0), mx.where(cand, full, 0)).item()  # exactos en los candidatos
    assert (two.argmax(-1) == full.argmax(-1)).mean().item() >= 0.98


def test_sampling_distribution_is_valid():
    logits = mx.random.normal((3, 1000), key=mx.random.key(7)) * 3
    for sp in (Sampling(temperature=0.7, top_k=20, top_p=0.9), Sampling(temperature=1.0, top_p=0.5),
               Sampling(temperature=1.0, min_p=0.1)):
        idx, lp = Engine._dist(logits, sp)
        p = mx.exp(lp)
        assert mx.allclose(p.sum(-1), mx.ones((3,)), atol=1e-4).item()
        if sp.top_k:
            assert idx.shape == (3, sp.top_k)
