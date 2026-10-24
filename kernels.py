"""Kernels Metal para el MoE de decode (batch 1) de DeepSeek-V4.x con expertos MXFP4.

Formato de pesos = el de MLX mode="mxfp4" (y el de los checkpoints V4.1-Flash):
  w: uint32 [E, N, K/8]  -> 8 nibbles e2m1 por palabra, nibble bajo = elemento menor
  s: uint8  [E, N, K/32] -> escala e8m0, valor = 2^(s-127)

Dos dispatches por capa MoE (en vez de ~10-15 ops de MLX):
  moe_gate_up : h[slot] = silu(min(x·Wg, L)) * clamp(x·Wu, -L, L)   para los 6+1 expertos
  moe_down    : y = sum_slot score[slot] * (h[slot]·Wd)              reducción entre expertos dentro del kernel
"""
import mlx.core as mx

HEADER = r"""
// e2m1 -> half por bits: (s e e m) va a signo + bits 11..9 del half; el resultado es
// valor * 2^-14 (también para el subnormal 0.5), así que el 2^14 se pliega en la escala.
inline float dot8(uint w, thread const float* xv) {
  float acc = 0.0f;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint t = w >> (4 * j);
    uint bits = ((t & 0x00070007u) << 9) | ((t & 0x00080008u) << 12);
    float2 v = float2(as_type<half2>(bits));
    acc = fma(v.x, xv[j], acc);
    acc = fma(v.y, xv[j + 4], acc);
  }
  return acc;
}
inline float dot32(uint4 w, thread const float* xv) {
  return dot8(w.x, xv) + dot8(w.y, xv + 8) + dot8(w.z, xv + 16) + dot8(w.w, xv + 24);
}
// 2^(s-127) * 2^14, válido para s <= 240
inline float e8m0_scale(uint8_t s) { return as_type<float>((uint(s) + 14u) << 23); }
"""

GATE_UP_SRC = r"""
  constexpr int KG = H / 32;                  // grupos (uint4) por fila
  const uint lane = thread_index_in_simdgroup;
  const uint sgi = simdgroup_index_in_threadgroup;
  const uint slot = threadgroup_position_in_grid.y;
  const uint row0 = (threadgroup_position_in_grid.x * NSG + sgi) * ROWS;
  const int e = idx[slot];
  const size_t base = (size_t)e * I + row0;
  const device uint4* g4 = (const device uint4*)wg + base * KG;
  const device uint4* u4 = (const device uint4*)wu + base * KG;
  const device uint8_t* gs = sg + base * KG;
  const device uint8_t* us = su + base * KG;

  float accg[ROWS], accu[ROWS];
  #pragma unroll
  for (int r = 0; r < ROWS; ++r) { accg[r] = 0.0f; accu[r] = 0.0f; }

  for (int g = lane; g < KG; g += 32) {
    float xv[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) xv[i] = float(x[g * 32 + i]);
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      uint4 wq = g4[r * KG + g];
      uint4 uq = u4[r * KG + g];
      accg[r] = fma(e8m0_scale(gs[r * KG + g]), dot32(wq, xv), accg[r]);
      accu[r] = fma(e8m0_scale(us[r * KG + g]), dot32(uq, xv), accu[r]);
    }
  }
  #pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float gv = simd_sum(accg[r]);
    float uv = simd_sum(accu[r]);
    if (lane == 0) {
      gv = min(gv, LIMIT);
      uv = clamp(uv, -LIMIT, LIMIT);
      h[slot * I + row0 + r] = gv / (1.0f + exp(-gv)) * uv;
    }
  }
"""

DOWN_SRC = r"""
  constexpr int KG = I / 32;                  // grupos por fila de down
  constexpr int TASKS = NE * KG;              // (experto, grupo) concatenados: la suma entre expertos sale gratis
  const uint lane = thread_index_in_simdgroup;
  const uint sgi = simdgroup_index_in_threadgroup;
  const uint row0 = (threadgroup_position_in_grid.x * NSG + sgi) * ROWS;
  const device uint4* d4 = (const device uint4*)wd;

  float acc[ROWS];
  #pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

  for (int t = lane; t < TASKS; t += 32) {
    const int slot = t / KG;
    const int g = t - slot * KG;
    const size_t rbase = (size_t)idx[slot] * H + row0;
    const float w = scores[slot];
    float xv[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) xv[i] = h[slot * I + g * 32 + i] * w;
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const size_t off = (rbase + r) * KG + g;
      acc[r] = fma(e8m0_scale(sd[off]), dot32(d4[off], xv), acc[r]);
    }
  }
  #pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = simd_sum(acc[r]);
    if (lane == 0) y[row0 + r] = T(v);
  }
"""

_gate_up_cache = {}


def _gate_up(limit):
    # metal_kernel no admite floats como template: el límite de SwiGLU va en el fuente
    if limit not in _gate_up_cache:
        _gate_up_cache[limit] = mx.fast.metal_kernel(
            name="ds_moe_gate_up_mxfp4",
            input_names=["x", "wg", "sg", "wu", "su", "idx"],
            output_names=["h"],
            source=GATE_UP_SRC.replace("LIMIT", f"{float(limit)!r}f"),
            header=HEADER,
        )
    return _gate_up_cache[limit]

_down = mx.fast.metal_kernel(
    name="ds_moe_down_mxfp4",
    input_names=["h", "wd", "sd", "idx", "scores"],
    output_names=["y"],
    source=DOWN_SRC,
    header=HEADER,
)


def moe_decode(x, wg, sg, wu, su, wd, sd, idx, scores, *, limit=10.0,
               nsg=1, rows_gu=4, rows_down=2):
    """x: [H] (bf16/f16/f32), idx: int32 [NE] (routed + shared), scores: f32 [NE]."""
    H = x.shape[-1]
    I = wg.shape[1]
    NE = idx.shape[0]
    assert I % (nsg * rows_gu) == 0 and H % (nsg * rows_down) == 0
    h = _gate_up(limit)(
        inputs=[x, wg, sg, wu, su, idx],
        template=[("T", x.dtype), ("H", H), ("I", I), ("NSG", nsg), ("ROWS", rows_gu)],
        grid=(32 * nsg * (I // (nsg * rows_gu)), NE, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(NE, I)],
        output_dtypes=[mx.float32],
    )[0]
    return _down(
        inputs=[h, wd, sd, idx, scores],
        template=[("T", x.dtype), ("H", H), ("I", I), ("NE", NE), ("NSG", nsg), ("ROWS", rows_down)],
        grid=(32 * nsg * (H // (nsg * rows_down)), 1, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(H,)],
        output_dtypes=[x.dtype],
    )[0]
