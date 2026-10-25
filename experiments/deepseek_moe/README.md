# Experimento: MoE de DeepSeek‑V4.1‑Flash en Metal

Origen del proyecto. Kernels Metal (vía `mx.fast.metal_kernel`) para el bloque MoE de decode de
**DeepSeek‑V4.1‑Flash** (H=5120, I=2304, 384 expertos, top‑6 + 1 compartido, expertos en MXFP4: e2m1 +
escala e8m0 por bloque de 32, SwiGLU con límite 10), medidos con pesos sintéticos de las formas reales.

```bash
.venv/bin/python experiments/deepseek_moe/bench_moe.py   # exactitud + ms/capa frente a MLX
.venv/bin/python bw_probe.py                             # techo de ancho de banda de la GPU
```

- `kernels.py`: MoE de decode en 2 dispatches. `moe_gate_up` calcula gate y up de los 7 expertos leyendo
  `x` una vez, con el SwiGLU limitado ya aplicado. `moe_down` hace el down y la suma ponderada en el mismo
  kernel, concatenando los pares (experto, grupo). El e2m1 se decodifica por bits a `half2`, sin tablas.
- `bench_moe.py`: un eval por "token" atravesando 8 capas distintas.

## Resultado (MacBook Air M4, 24 GB)

| | ms/capa MoE | GB/s |
|---|---|---|
| techo de lectura medido (`bw_probe.py`) | — | ~99 |
| MLX estilo mlx‑lm (3 gather_qmm + compartido) | 1.36–1.47 | 90–97 |
| kernel propio (2 dispatches) | 1.35–1.47 | 90–98 |

**Empate con MLX** dentro del ruido (±3 %). Con FP4 y batch 1 el MoE está limitado por memoria y MLX ya
llega al ~95 % del techo. Exactitud frente a fp32: 4e‑7 (fp32) y 2,4e‑3 en bf16, mejor que MLX (6,3e‑3),
porque el intermedio se queda en fp32.

DeepSeek‑V4.1‑Flash pesa **510 GB** (~552B parámetros, 16B activos en decode): no cabe en un Mac de
24 GB. Incluso V4‑Flash (284B) necesita Macs de 128 GB; el motor de referencia en Metal es
[antirez/ds4](https://github.com/antirez/ds4). Por eso el proyecto pasó a acelerar un modelo que sí cabe:
Qwen3.5‑9B.
