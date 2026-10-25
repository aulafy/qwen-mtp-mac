"""Tramo central de GatedDeltaNet (Qwen3.5) fusionado para decode/verificación (S <= 16 tokens, B = 1).

Partes adaptadas de mlx-lm (https://github.com/ml-explore/mlx-lm), Copyright © 2023 Apple Inc., licencia MIT.

mlx-lm hace ~20 operaciones pequeñas por capa entre las proyecciones (conv1d, silu, normas de q/k,
compuertas, delta rule, norma con compuerta). Aquí son 3: concatenar la ventana de la convolución y
dos kernels Metal:
  gdn_core       conv1d + silu, rms_norm de q y k, g = exp(-exp(A_log)·softplus(a + dt_bias)),
                 beta = sigmoid(b), delta rule (estado en registros) e instantáneas para deshacer
  gdn_norm_gate  rms_norm(y)·w · silu(z) por cabeza (necesita los 128 valores de cada cabeza)
Redondea a bf16 en los mismos puntos que mlx-lm (salida de conv, silu, normas, compuertas, y) para dar
prácticamente lo mismo que la ruta original; solo cambia el orden de algunas sumas.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5

import mtp as M

# Mismas fórmulas que los kernels de MLX en bf16, redondeando tras cada operación (unary_ops.h,
# binary_ops.h, utils.h): coinciden al 100 % con mx.sigmoid, nn.silu y nn.softplus en los 65.536 bf16.
BF16_MATH = r"""
inline float R(float x) { return static_cast<float>(static_cast<bfloat16_t>(x)); }
inline float sigmoid_bf(float x) {
  const float y = R(1.0f / R(1.0f + R(metal::exp(metal::abs(x)))));
  return x < 0 ? y : R(1.0f - y);
}
inline float silu_bf(float x) { return R(x * sigmoid_bf(x)); }
inline float log1p_bf(float x) {
  const float xp1 = 1.0f + x;
  return xp1 == 1.0f ? x : R(x * (metal::log(xp1) / (xp1 - 1.0f)));
}
inline float softplus_bf(float x) {  // logaddexp(x, 0)
  const float hi = metal::max(x, 0.0f), lo = metal::min(x, 0.0f);
  return R(hi + log1p_bf(R(metal::exp(R(lo - hi)))));
}
inline float silu_f32(float x) {      // nn.silu en float32 (_precise_swiglu)
  const float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
  return x * (x < 0 ? y : 1.0f - y);
}
"""

CORE_SRC = r"""
  // grid (32, Dv, Hv), threadgroup (32, 4, 1); B = 1
  const uint lane = thread_position_in_threadgroup.x;
  const uint dv = thread_position_in_grid.y;
  const uint hv = thread_position_in_grid.z;
  const uint hk = hv / (Hv / Hk);
  constexpr int NP = Dk / 32;                  // elementos de dk por hilo
  constexpr int C = 2 * Hk * Dk + Hv * Dv;     // canales de la convolución (q, k, v)
  const int cq = hk * Dk + NP * lane;          // canal del primer q de este hilo
  const int ck = Hk * Dk + hk * Dk + NP * lane;
  const int cv = 2 * Hk * Dk + hv * Dv + dv;

  const float A = -exp(float(A_log[hv]));
  const float dtb = float(dt_bias[hv]);
  const float kscale = R(metal::precise::rsqrt(float(Dk)));  // escalar de Python -> bf16 en mlx-lm
  const size_t srow = ((size_t)hv * Dv + dv) * Dk + NP * lane;

  float st[NP];
  for (int i = 0; i < NP; ++i) st[i] = state_in[srow + i];

  for (int t = 0; t < S; ++t) {
    // conv1d causal de KC tomas sobre la ventana [estado previo, tokens] + silu (redondeos como mlx-lm)
    float q[NP], k[NP];
    float qs = 0.0f, ks = 0.0f;
    for (int i = 0; i < NP; ++i) {
      float aq = 0.0f, ak = 0.0f;
      for (int j = 0; j < KC; ++j) {
        aq += float(conv_w[(cq + i) * KC + j]) * float(win[(t + j) * C + cq + i]);
        ak += float(conv_w[(ck + i) * KC + j]) * float(win[(t + j) * C + ck + i]);
      }
      q[i] = silu_bf(R(aq));
      k[i] = silu_bf(R(ak));
      qs += q[i] * q[i];
      ks += k[i] * k[i];
    }
    float av = 0.0f;
    for (int j = 0; j < KC; ++j) av += float(conv_w[cv * KC + j]) * float(win[(t + j) * C + cv]);
    const float v = silu_bf(R(av));
    // q = rms_norm(q)/Dk, k = rms_norm(k)·Dk^-1/2 (rms_norm devuelve bf16; 1/Dk es exacto)
    const float rq = metal::precise::rsqrt(simd_sum(qs) / Dk + 1e-6f);
    const float rk = metal::precise::rsqrt(simd_sum(ks) / Dk + 1e-6f);
    for (int i = 0; i < NP; ++i) {
      q[i] = R(q[i] * rq) / Dk;
      k[i] = R(R(k[i] * rk) * kscale);
    }
    // g = exp(-exp(A_log)·softplus(a + dt_bias)) con a + dt_bias y softplus en bf16; beta = sigmoid(b)
    const float sp = softplus_bf(R(float(a[t * Hv + hv]) + dtb));
    const float g = metal::exp(A * sp);
    const float beta = sigmoid_bf(float(b[t * Hv + hv]));

    float kv_mem = 0.0f;
    for (int i = 0; i < NP; ++i) {
      st[i] *= g;
      kv_mem += st[i] * k[i];
    }
    kv_mem = simd_sum(kv_mem);
    const float delta = (v - kv_mem) * beta;
    float o = 0.0f;
    for (int i = 0; i < NP; ++i) {
      st[i] += k[i] * delta;
      o += st[i] * q[i];
    }
    o = simd_sum(o);
    if (lane == 0) y[((size_t)t * Hv + hv) * Dv + dv] = R(o);
    if (SNAP && t < S - 1) {
      for (int i = 0; i < NP; ++i) state_mid[(size_t)t * Hv * Dv * Dk + srow + i] = st[i];
    }
  }
  for (int i = 0; i < NP; ++i) state_out[srow + i] = st[i];
"""

NORM_SRC = r"""
  // una simdgroup por (token, cabeza): rms_norm(y)·w · silu(z)
  const uint lane = thread_position_in_threadgroup.x;
  const uint row = thread_position_in_grid.y;      // t * Hv + h
  constexpr int NP = Dv / 32;
  const size_t base = (size_t)row * Dv + NP * lane;
  float yv[NP];
  float ss = 0.0f;
  for (int i = 0; i < NP; ++i) { yv[i] = y[base + i]; ss += yv[i] * yv[i]; }
  const float r = metal::precise::rsqrt(simd_sum(ss) / Dv + EPS);
  for (int i = 0; i < NP; ++i) {
    const float zz = float(z[base + i]);
    const float xn = R(float(w[NP * lane + i]) * R(yv[i] * r));      // = mx.fast.rms_norm (bf16)
    out[base + i] = T(xn * silu_f32(zz));                             // _precise_swiglu en float32
  }
"""

_core = mx.fast.metal_kernel(
    name="gdn_core",
    input_names=["win", "conv_w", "a", "b", "A_log", "dt_bias", "state_in"],
    output_names=["y", "state_out", "state_mid"],
    source=CORE_SRC,
    header=BF16_MATH,
)
_norm_cache = {}


def _norm(eps):
    if eps not in _norm_cache:
        _norm_cache[eps] = mx.fast.metal_kernel(
            name="gdn_norm_gate", input_names=["y", "z", "w"], output_names=["out"],
            source=NORM_SRC.replace("EPS", f"{float(eps)!r}f"), header=BF16_MATH)
    return _norm_cache[eps]


def gdn_middle(self, qkv, z, a, b, conv_state, state, snap):
    """qkv [1,S,C], z [1,S,Hv,Dv], a/b [1,S,Hv] -> (out [1,S,Hv*Dv], ventana conv, estado, instantáneas)."""
    S = qkv.shape[1]
    Hk, Hv, Dk, Dv = self.num_k_heads, self.num_v_heads, self.head_k_dim, self.head_v_dim
    win = mx.concatenate([conv_state, qkv], axis=1)
    y, st, mids = _core(
        inputs=[win, self.conv1d.weight, a, b, self.A_log, self.dt_bias, state],
        template=[("S", S), ("Hk", Hk), ("Hv", Hv), ("Dk", Dk), ("Dv", Dv),
                  ("KC", self.conv_kernel_size), ("SNAP", bool(snap))],
        grid=(32, Dv, Hv), threadgroup=(32, 4, 1),
        output_shapes=[(S, Hv, Dv), state.shape, (max(S - 1, 1), *state.shape)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    out = _norm(self.norm.eps)(
        inputs=[y, z, self.norm.weight], template=[("T", qkv.dtype), ("Dv", Dv)],
        grid=(32, S * Hv, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1, S, Hv * Dv)], output_dtypes=[qkv.dtype],
    )[0]
    return out, win, st, mids


MAX_S = 16


def _call(self, inputs, mask=None, cache=None):
    B, S, _ = inputs.shape
    if B != 1 or S > MAX_S or mask is not None or cache is None or cache.lengths is not None:
        return M._gdn_call(self, inputs, mask, cache)  # prefill y casos generales: ruta de mlx-lm
    qkv = self.in_proj_qkv(inputs)
    z = self.in_proj_z(inputs)
    a = self.in_proj_a(inputs)
    b = self.in_proj_b(inputs)
    n_keep = self.conv_kernel_size - 1
    conv_state = cache[0] if cache[0] is not None else mx.zeros((1, n_keep, self.conv_dim), dtype=inputs.dtype)
    state = cache[1] if cache[1] is not None else mx.zeros(
        (1, self.num_v_heads, self.head_v_dim, self.head_k_dim), dtype=mx.float32)
    snap = getattr(cache, "snap", False) and S > 1
    out, win, st, mids = gdn_middle(self, qkv, z, a, b, conv_state, state, snap)
    cache[0] = win[:, -n_keep:, :]
    cache[1] = st
    cache.advance(S)
    if snap:
        cache.rollback = (win, mids, n_keep)
    return self.out_proj(out)


def install():
    """Sustituye GatedDeltaNet.__call__ (decode/verificación) por la ruta fusionada."""
    qwen3_5.GatedDeltaNet.__call__ = _call


def uninstall():
    qwen3_5.GatedDeltaNet.__call__ = M._gdn_call
