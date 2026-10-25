"""GEMV NVFP4 para pocas filas (M = 2..4): lee cada peso UNA vez para todas las filas de x.

El quantized_matmul de MLX en modo nvfp4 crece con M (M=3: +13 %, M=4: +47 %, M=8: x2.8 sobre M=1),
y eso es justo lo que paga la verificación especulativa de 3-4 tokens. Aquí el coste debería ser
plano en M mientras se esté limitado por memoria.

Formato (= MLX mode="nvfp4", y el de Ollama qwen3.5:9b-mlx):
  w: uint32 [N, K/8]  -> 8 nibbles e2m1 por palabra, nibble bajo = elemento menor
  s: uint8  [N, K/16] -> escala fp8 e4m3 por grupo de 16
"""
import mlx.core as mx
import mlx.nn as nn

HEADER = r"""
// e2m1 -> half por bits (valor * 2^-14) y e4m3 -> half por bits (valor * 2^-8): 2^22 va en la escala.
// Una palabra (8 nibbles) -> dos float4 en el orden de los elementos 0..3 y 4..7.
inline void deq8(uint w, thread float4& lo, thread float4& hi) {
  float2 v0 = float2(as_type<half2>(((w & 0x00070007u) << 9) | ((w & 0x00080008u) << 12)));
  uint t1 = w >> 4, t2 = w >> 8, t3 = w >> 12;
  float2 v1 = float2(as_type<half2>(((t1 & 0x00070007u) << 9) | ((t1 & 0x00080008u) << 12)));
  float2 v2 = float2(as_type<half2>(((t2 & 0x00070007u) << 9) | ((t2 & 0x00080008u) << 12)));
  float2 v3 = float2(as_type<half2>(((t3 & 0x00070007u) << 9) | ((t3 & 0x00080008u) << 12)));
  lo = float4(v0.x, v1.x, v2.x, v3.x);
  hi = float4(v0.y, v1.y, v2.y, v3.y);
}
inline float e4m3_scale(uint8_t s) {
  return float(as_type<half>(ushort(uint(s & 0x7f) << 7))) * 4194304.0f;
}
"""

SRC = r"""
  constexpr int KG = K / 16;                  // grupos de 16 por fila
  const uint lane = thread_index_in_simdgroup;
  const uint sgi = simdgroup_index_in_threadgroup;
  const uint row0 = (threadgroup_position_in_grid.x * NSG + sgi) * ROWS;
  const device uint2* w2 = (const device uint2*)w + (size_t)row0 * KG;
  const device uint8_t* sp = s + (size_t)row0 * KG;

  float acc[M][ROWS];
  #pragma unroll
  for (int m = 0; m < M; ++m)
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[m][r] = 0.0f;

  for (int g = lane; g < KG; g += 32) {
    float4 xv[M][4];
    #pragma unroll
    for (int m = 0; m < M; ++m) {
      const device T* xp = x + m * K + g * 16;
      #pragma unroll
      for (int i = 0; i < 4; ++i)
        xv[m][i] = float4(float(xp[4 * i]), float(xp[4 * i + 1]), float(xp[4 * i + 2]), float(xp[4 * i + 3]));
    }
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      uint2 wq = w2[r * KG + g];
      float sc = e4m3_scale(sp[r * KG + g]);
      float4 w0, w1, w2v, w3;
      deq8(wq.x, w0, w1);
      deq8(wq.y, w2v, w3);
      #pragma unroll
      for (int m = 0; m < M; ++m) {
        float d = (dot(w0, xv[m][0]) + dot(w1, xv[m][1])) + (dot(w2v, xv[m][2]) + dot(w3, xv[m][3]));
        acc[m][r] = fma(sc, d, acc[m][r]);
      }
    }
  }
  #pragma unroll
  for (int m = 0; m < M; ++m)
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float v = simd_sum(acc[m][r]);
      if (lane == 0) y[m * N + row0 + r] = T(v);
    }
"""

_kernel = mx.fast.metal_kernel(
    name="qmv_nvfp4_multi",
    input_names=["x", "w", "s"],
    output_names=["y"],
    source=SRC,
    header=HEADER,
)


def qmv_nvfp4(x, w, s, rows=None, nsg=1):
    """x: [..., K] con prod(...) = M filas; w: uint32 [N, K/8]; s: uint8 [N, K/16] -> [..., N]."""
    K = x.shape[-1]
    N = w.shape[0]
    M = x.size // K
    rows = rows or (8 if M <= 3 else 4)  # medido en M4: M=3 -> r8s1, M=4 -> r4s1
    assert N % (rows * nsg) == 0 and K % 16 == 0
    y = _kernel(
        inputs=[x.reshape(M, K), w, s],
        template=[("T", x.dtype), ("M", M), ("K", K), ("N", N), ("ROWS", rows), ("NSG", nsg)],
        grid=(32 * nsg * (N // (rows * nsg)), 1, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )[0]
    return y.reshape(*x.shape[:-1], N)


_orig_call = nn.QuantizedLinear.__call__
MIN_ROWS, MAX_ROWS = 3, 4  # con 1-2 filas MLX ya va al ancho de banda


def _call(self, x):
    rows = x.size // x.shape[-1]
    if (getattr(self, "mode", "affine") == "nvfp4" and MIN_ROWS <= rows <= MAX_ROWS
            and "bias" not in self):
        return qmv_nvfp4(x, self["weight"], self["scales"])
    return _orig_call(self, x)


def install(min_rows=3, max_rows=4):
    """Sustituye QuantizedLinear (nvfp4) por este kernel cuando x tiene entre min_rows y max_rows filas."""
    global MIN_ROWS, MAX_ROWS
    MIN_ROWS, MAX_ROWS = min_rows, max_rows
    nn.QuantizedLinear.__call__ = _call
