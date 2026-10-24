"""Techo de ancho de banda de lectura alcanzable en esta GPU (lectura pura uint4)."""
import statistics
import time

import mlx.core as mx

SRC = r"""
  const device uint4* p = (const device uint4*)a;
  uint4 acc = 0;
  for (uint i = thread_position_in_grid.x; i < N; i += threads_per_grid.x) acc ^= p[i];
  out[thread_position_in_grid.x] = acc.x ^ acc.y ^ acc.z ^ acc.w;
"""
k = mx.fast.metal_kernel(name="bw_read", input_names=["a"], output_names=["out"], source=SRC)

nbytes = 1 << 30
a = mx.random.randint(0, 2**31 - 1, (nbytes // 4,)).astype(mx.uint32)
mx.eval(a)
for threads in (1 << 14, 1 << 16, 1 << 18, 1 << 20):
    for tg in (256, 1024):
        def run():
            return k(inputs=[a], template=[("N", nbytes // 16)], grid=(threads, 1, 1),
                     threadgroup=(tg, 1, 1), output_shapes=[(threads,)], output_dtypes=[mx.uint32])[0]
        mx.eval(run())
        ts = []
        for _ in range(7):
            t0 = time.perf_counter(); mx.eval(run()); ts.append(time.perf_counter() - t0)
        print(f"threads {threads:8d} tg {tg:5d}: {nbytes/statistics.median(ts)/1e9:6.1f} GB/s")
