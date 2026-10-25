"""GatedDeltaNet de decode con instantánea del estado, en un solo dispatch Metal.

Partes adaptadas de mlx-lm (https://github.com/ml-explore/mlx-lm), Copyright © 2023 Apple Inc., licencia MIT.

Variante del kernel de mlx-lm (gated_delta.py, sin máscara, gating escalar) que, además del estado
final, escribe el estado tras cada uno de los primeros T-1 tokens. Es lo que necesita la verificación
especulativa: procesar [t, d1, ..., dk] de una vez y poder volver al estado tras cualquier prefijo.
"""
import mlx.core as mx

SRC = r"""
  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);
  constexpr int n_per_t = Dk / 32;

  auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
  y += b_idx * T * Hv * Dv + hv_idx * Dv;

  auto dk_idx = thread_position_in_threadgroup.x;
  auto dv_idx = thread_position_in_grid.y;

  auto i_state = state_in + (n * Dv + dv_idx) * Dk;
  auto o_state = state_out + (n * Dv + dv_idx) * Dk;
  auto m_state = state_mid + (n * Dv + dv_idx) * Dk;

  float state[n_per_t];
  for (int i = 0; i < n_per_t; ++i) state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);

  auto g_ = g + b_idx * T * Hv;
  auto beta_ = beta + b_idx * T * Hv;

  for (int t = 0; t < T; ++t) {
    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = state[i] * g_[hv_idx];
      kv_mem += state[i] * k_[s_idx];
    }
    kv_mem = simd_sum(kv_mem);
    auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = state[i] + k_[s_idx] * delta;
      out += state[i] * q_[s_idx];
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0) y[dv_idx] = static_cast<InT>(out);
    if (t < T - 1) {
      for (int i = 0; i < n_per_t; ++i) m_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
      m_state += B * Hv * Dv * Dk;
    }
    q_ += Hk * Dk;
    k_ += Hk * Dk;
    v_ += Hv * Dv;
    y += Hv * Dv;
    g_ += Hv;
    beta_ += Hv;
  }
  for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
"""

_kernel = mx.fast.metal_kernel(
    name="gated_delta_snapshot",
    input_names=["q", "k", "v", "g", "beta", "state_in"],
    output_names=["y", "state_out", "state_mid"],
    source=SRC,
)


def gated_delta_snapshot(q, k, v, g, beta, state):
    """q, k: [B,T,Hk,Dk]; v: [B,T,Hv,Dv]; g, beta: [B,T,Hv]; state: [B,Hv,Dv,Dk].
    Devuelve (y, estado final, estados tras 1..T-1 tokens: [T-1, B, Hv, Dv, Dk])."""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    return _kernel(
        inputs=[q, k, v, g, beta, state],
        template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv),
                  ("Hk", Hk), ("Hv", Hv), ("T", T), ("B", B)],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape, (max(T - 1, 1), *state.shape)],
        output_dtypes=[q.dtype, state.dtype, state.dtype],
    )
