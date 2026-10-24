# ds-metal

Kernels Metal (vía `mx.fast.metal_kernel` de MLX) para el decode de DeepSeek‑V4.x en Apple Silicon.
Primer objetivo: el bloque MoE de **DeepSeek‑V4.1‑Flash** (H=5120, I=2304, 384 expertos, top‑6 + 1 compartido,
expertos en MXFP4: e2m1 + escala e8m0 por bloque de 32, SwiGLU con límite 10).

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python mlx==0.31.2
.venv/bin/python bench_moe.py      # exactitud + ms/capa frente a MLX
.venv/bin/python bw_probe.py       # techo de ancho de banda de la GPU
```

## Qué hay

- `kernels.py`: MoE de decode en 2 dispatches.
  - `moe_gate_up`: gate y up de los 7 expertos leyendo `x` una vez, con el SwiGLU limitado ya aplicado.
  - `moe_down`: down de los 7 expertos y suma ponderada dentro del mismo kernel. Concatena los pares
    (experto, grupo), así que la reducción entre expertos sale gratis.
  - El e2m1 se decodifica por bits a `half2`, con el 2^14 plegado en la escala. No usa tablas.
- `bench_moe.py`: pesos sintéticos MXFP4 con las formas reales. No hace falta el checkpoint para medir.
  Hace un eval por "token" atravesando 8 capas distintas.

## Resultados (Mac M4 base, 24 GB, 25‑oct‑2026)

| | ms/capa MoE | GB/s |
|---|---|---|
| techo de lectura medido (`bw_probe.py`) | — | ~99 |
| MLX estilo mlx‑lm (3 gather_qmm + compartido) | 1.36–1.47 | 90–97 |
| **ds-metal** (2 dispatches) | 1.35–1.47 | 90–98 |

- Exactitud frente a la referencia fp32 (dequantize): fp32 4e‑7; bf16 2.4e‑3, mejor que MLX (6.3e‑3)
  porque `h` se queda en fp32.
- **Velocidad: empate con MLX**, dentro del ruido (±3 %). Con FP4 y batch 1 el MoE ya está limitado por
  memoria, y MLX llega al ~95 % del techo real. Ningún kernel puede ganar aquí más de un ~3–5 %.

## Dónde está la ganancia de verdad en Mac

1. **Overhead por dispatch**: medido en ~3.4–5 µs por op en MLX. Las partes nuevas de V4.1 son decenas
   o cientos de ops pequeñas por capa: mHC con sinkhorn de 20 iteraciones, atención dispersa comprimida
   con indexador y engram. Fusionarlas en un kernel puede ahorrar del orden de 10–20 ms por token.
2. **Menos bytes por token**: MTP/especulativo (V4.1 trae 3 capas nextn y DSpark) o expertos a menos bits.

## Qwen3.5‑9B más rápido (el que ya tienes en Ollama)

Reutiliza los pesos NVFP4 de `qwen3.5:9b-mlx` de Ollama sin descargarlos otra vez (`qwen_ollama.py`).
Solo se bajan los ~0,5 GB de la cabeza MTP oficial, con peticiones por rangos (`fetch_mtp.py`).

```bash
.venv/bin/python fetch_mtp.py
.venv/bin/python mtp.py --lm-head 6 --mtp-quant nvfp4 --draft-vocab 248320 --draft-bits 4
```

| Configuración (M4 24 GB, greedy, 4 prompts × 256 tokens) | tok/s |
|---|---|
| Ollama `qwen3.5:9b-mlx` | 15,2 |
| MLX con los mismos pesos | 15,0–15,4 |
| + `lm_head` cuantizado (Ollama lo deja en bf16: 2 GB por token) | 17,5–18,7 (8 bits) · 20,2 (4 bits) |
| + decodificación especulativa con la cabeza MTP nativa + kernel `gdn_kernel.py` | **22–26 (media 24,0)** |

- `lm_head`: a 8 bits coincide en el 98,7 % de los top‑1 (KL 0,0002); a 6 bits igual (KL 0,0005); a 4 bits
  baja al 97 % (KL 0,004).
- MTP: la cabeza propone 1 token y el modelo verifica 2 en una sola pasada. La aceptación va del 69 al 85 %.
  Con temperatura 0 la salida coincide token a token con el greedy normal hasta que un empate numérico
  de bf16 la desvía. Sigue siendo texto del modelo.
- `gdn_kernel.py`: GatedDeltaNet en un solo dispatch Metal que guarda el estado intermedio para poder
  deshacer si el borrador falla. Es idéntico bit a bit a dos llamadas del kernel de mlx-lm y deja la
  verificación de 2 tokens en 58 ms, frente a 56 ms para 1. Sin él, partir la llamada en dos se comía
  la ganancia.

## Límite de hardware

DeepSeek‑V4.1‑Flash pesa **510 GB** (unos 552B parámetros, 16B activos en decode). No cabe en 24 GB de RAM
ni en el disco. Incluso V4‑Flash (284B) necesita Macs de 128 GB (el motor de referencia en Metal es
[antirez/ds4](https://github.com/antirez/ds4)). En este M4 los kernels se desarrollan y validan con formas
reales y pesos sintéticos. Para tokens/s de extremo a extremo hace falta un modelo que quepa en 24 GB.
