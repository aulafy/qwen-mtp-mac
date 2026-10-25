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
.venv/bin/python fetch_mtp.py          # cabeza MTP (~0,5 GB)
.venv/bin/python build_draft_vocab.py  # ids frecuentes para el lm_head de borrador (corpus = carpeta padre)
.venv/bin/python mtp.py                # k=2, lm_head en dos fases, borrador 64k ids a 4 bits
.venv/bin/python bench_ab.py           # banco intercalado base / k1 / k2
```

| Configuración (M4 24 GB, greedy, 4 prompts × 256 tokens) | tok/s |
|---|---|
| Ollama `qwen3.5:9b-mlx` | 15,2 |
| MLX con los mismos pesos | 15,0–15,4 |
| + `lm_head` cuantizado (Ollama lo deja en bf16: 2 GB por token) | 17,5–18,7 (8 bits) · 20,2 (4 bits) |
| + decodificación especulativa con la cabeza MTP nativa + kernel `gdn_kernel.py` | **22–26 (media 24,0)** |

### 2 borradores por ciclo (k=2)

El equipo es un MacBook Air M4 (Mac16,13), sin ventilador. Con carga sostenida la GPU baja de frecuencia:
la base pasa de ~18,7 a ~15 tok/s en pocos minutos. Por eso `bench_ab.py` intercala las configuraciones
prompt a prompt. Resultado con el equipo ya caliente (2 rondas × 4 prompts × 128 tokens):

| Configuración | tok/s | vs base | tokens/ciclo |
|---|---|---|---|
| base (mlx-lm) | 15,1 | 1,00× | — |
| MTP k=1 | 19,6 | 1,30× | 1,72 |
| **MTP k=2 + borrador 64k ids** | **21,3** | **1,41×** | 2,13 |
| MTP k=2, borrador con vocabulario completo | 20,1 | 1,33× | 2,18 |

En frío el primer prompt llega a 27–29 tok/s con k=2. k=3 no compensa: el tercer borrador acierta solo
el 16–40 % de las veces.

Qué hizo falta para que k=2 gane a k=1:
- **`qmv_kernel.py`**: GEMV NVFP4 para 3–4 filas que lee cada peso una vez. El `quantized_matmul` de MLX
  crece con M (M=3 +16 %, M=4 +53 %). El nuestro, con productos `float4`, se queda plano y es exacto
  (logits idénticos bit a bit a MLX): 3 filas en 294 µs frente a 339, 4 filas en 305 frente a 447 (capa
  de 12288×4096). Verificar 3 tokens pasa de 69 a 61 ms y 4 tokens de 89 a 66 ms. La primera versión,
  con cadenas de FMA dependientes, era más lenta que MLX (505 µs).
- **`gdn_kernel.py`** guarda el estado tras cada token para volver a cualquier prefijo.
- **Borrador barato**: el `lm_head` de borrador solo puntúa los 64k ids más frecuentes de un corpus
  local, mezcla normalizada de todo (código/JSON) y solo prosa. Cubre el 93–98 % de lo que genera el
  modelo, frente al 73–90 % del top‑32k con el corpus solo de código. Cada borrador baja de ~8 ms.
- El segundo borrador encadena el hidden pre-norm de la propia MTP (pre y post-norm aciertan igual).

- `lm_head`: a 8 bits coincide en el 98,7 % de los top‑1 (KL 0,0002); a 6 bits igual (KL 0,0005); a 4 bits
  baja al 97 % (KL 0,004).
- MTP: la cabeza propone 1 token y el modelo verifica 2 en una sola pasada. La aceptación va del 69 al 85 %.
  Con temperatura 0 la salida coincide token a token con el greedy normal hasta que un empate numérico
  de bf16 la desvía. Sigue siendo texto del modelo.
- `gdn_kernel.py`: GatedDeltaNet en un solo dispatch Metal que guarda el estado intermedio para poder
  deshacer si el borrador falla. Es idéntico bit a bit a dos llamadas del kernel de mlx-lm y deja la
  verificación de 2 tokens en 58 ms, frente a 56 ms para 1. Sin él, partir la llamada en dos se comía
  la ganancia.

### Lo que dio y lo que no (26‑oct, ideas del OKC)

- **Ganó: command buffers de MLX (`tuning.py`).** Con `MLX_MAX_OPS_PER_BUFFER=10`, la GPU empieza antes en
  vez de esperar a que se cierre un buffer grande. La pasada de 1 token baja de 56 a 52 ms y la de 3 tokens
  de 70 a 62. En el banco intercalado, MTP k=2 sube de 27,5 a **28,8 tok/s (1,47× la base)**; la base no
  cambia porque mlx-lm ya solapa CPU y GPU. Con el valor por defecto quedaba oculta la ganancia del GEMV
  multi-fila con 3 tokens. Hay que importar `tuning` antes que `mlx`, porque el valor se lee al crear el
  dispositivo.
- **No ganó: GatedDeltaNet fusionada (`gdn_fused.py`, desactivada).** Reduce las ~20 operaciones del
  tramo central a 3 y es **idéntica bit a bit** a mlx-lm: reproduce sus redondeos bf16 y sus fórmulas de
  sigmoid, softplus y rms_norm, comprobado sobre los 65.536 valores bf16. Aislada, baja ese tramo de 4,0 a
  2,9 ms con 3 tokens, pero en la pasada completa no se nota (61,8 frente a 62,2 ms; k2 28,8 frente a
  28,5). Con buffers pequeños, esas operaciones ya no frenan a la GPU. Se activa con
  `gdn_fused.install()`.
- **Ganó: `lm_head` en dos fases (`lm_head2.py`, por defecto).** Al verificar 3 tokens, el `lm_head` de
  6 bits costaba 13,2 ms en MLX. Ahora un grueso NVFP4 puntúa todo el vocabulario (6,4 ms con el GEMV
  multi-fila) y se recalculan con los pesos **bf16 originales** solo los 64 mejores candidatos: 7,4 ms en
  total. El top-1 del bf16 cae dentro del top-16 del grueso en el 100 % de las posiciones medidas, y su
  top-20 dentro del top-64. La salida es la del `lm_head` original (con 6 bits coincidía en el 99,2 %):
  en el mismo camino MTP da los mismos 200 tokens que el bf16 completo en los 4 prompts. Banco
  intercalado: k=2 **28,6 → 31,4 tok/s (1,65× la base)**. Para muestrear con top_k > 32 usa 4·top_k
  candidatos (256 si top_k = 0). Cuesta ~1,8 GB más de RAM, porque mantiene el `lm_head` bf16.
- **No ganaron: cabeza MTP a 8 bits ni k adaptativo** (`generate_mtp(..., tau=0.15)`, configuraciones
  `k2h8`, `k2ha` y `k3ha` de `bench_ab.py`). El k adaptativo verifica solo los borradores cuya
  probabilidad acumulada según la MTP supera τ. La MTP a 8 bits acierta algo más (2,19 tokens por ciclo
  frente a 2,14) y con k3ha llega a 2,30–2,35, pero cuestan más cálculo. En frío empatan (±2 %) y con el
  Air caliente pierden: 3 rondas, base 16,9 tok/s, k2h 27,1 (1,60×), k2h8 26,7, k3ha 25,1, k3ha8 25,5. En
  un equipo que baja de frecuencia, la configuración más ligera (k=2 fijo, MTP NVFP4) es la más robusta.
  Para acertar bastante más haría falta un árbol de borradores o reentrenar la cabeza MTP.
- `lab_gdn.py` separa en las 24 GDN el coste de las proyecciones (~10 ms, lectura de pesos) del tramo
  central (~3–4 ms).

## Chat (interfaz web mínima)

**Doble clic en `Chat Qwen.app`**: si el modelo no está en marcha lo carga (~5 s), y abre
`http://127.0.0.1:11435/chat` en el navegador. Para tenerlo en el Dock, arrastra la app allí. Si mueves la
carpeta, vuelve a crear la app con `./crear_app.sh`. Sin la app: `./chat.sh`.

- **Contexto** (arriba): 4K–64K tokens, con un medidor de lo usado. Si se llena, se olvidan los mensajes
  más antiguos (las instrucciones y la memoria se conservan) y la respuesta lo indica.
- **Memoria**: notas que se añaden a todos los chats (botón «Memoria», se guarda en `datos/memoria.md` y se
  puede editar a mano). El botón de la bombilla con + resume el chat actual y lo añade a la memoria.
- **Conversaciones**: se guardan solas en `datos/chats/*.json` y se retoman desde la barra lateral.
- **Archivos**: con el clip o arrastrándolos. Admite texto, código, CSV, JSON, PDF con texto y DOCX (hasta
  25 MB). Se convierten a texto y quedan guardados en el chat. Leerlos va a ~160 tok/s en el M4 (un archivo
  de 12K tokens tarda ~75 s la primera vez); las preguntas siguientes del mismo chat reutilizan lo leído.
- **Ajustes por chat**: instrucciones, temperatura, razonar (muestra el razonamiento plegado) y usar o no
  la memoria. Se puede detener una respuesta a medias.
- **RAM**: el servidor que arranca la app se cierra solo tras 30 min sin uso (`DS_METAL_IDLE`), o al pulsar
  «Apagar modelo». Log en `datos/servidor.log`.
- Solo responde a páginas de 127.0.0.1/localhost: una web cualquiera no puede leer tus chats ni usar el
  modelo.

## Servidor local (APIs de Ollama y OpenAI)

```bash
.venv/bin/python server.py              # http://127.0.0.1:11435, modelo "qwen3.5-mtp:9b"
.venv/bin/python server.py --port 11434 # en el puerto de Ollama (para Ollama antes)
.venv/bin/python test_server.py         # prueba con los clientes oficiales (pip install openai ollama)
```

- **Ollama**: `GET /`, `/api/version`, `/api/tags`, `/api/ps`; `POST /api/chat`, `/api/generate`
  (NDJSON en streaming o respuesta única), `/api/show`. Opciones: `temperature`, `top_k`, `top_p`, `min_p`,
  `num_predict`, `stop`, `seed`, y `think`, que devuelve `message.thinking` aparte.
- **OpenAI**: `GET /v1/models`; `POST /v1/chat/completions` y `/v1/completions` (SSE con `[DONE]`,
  `stream_options.include_usage`, `stop`, `max_tokens`/`max_completion_tokens`). El razonamiento sale en
  `reasoning_content` con `chat_template_kwargs: {"enable_thinking": true}`.
- En cualquier app: URL de Ollama `http://127.0.0.1:11435` o base de OpenAI `http://127.0.0.1:11435/v1`
  (cualquier api key). El nombre de modelo se ignora: solo hay uno.
- Por defecto usa los parámetros de Ollama para qwen3.5 (T=1, top_k 20, top_p 0,95). `--greedy` cambia el
  valor por defecto a T=0, y `--think` activa el razonamiento si la petición no lo indica.
- Con temperatura usa **muestreo especulativo exacto**: el borrador es determinista, así que se acepta con
  probabilidad p(d) y, si falla, se muestrea de p sin d. La distribución es la del modelo.
- **Caché entre turnos**: guarda el estado antes del prompt de generación (`<|im_start|>assistant…`) y en el
  turno siguiente solo procesa lo nuevo. Recorta las KV caches y restaura las GatedDeltaNet.
- Un único hilo es dueño del modelo (los streams de MLX son por hilo). Las peticiones simultáneas esperan
  turno y, si el cliente corta, la generación se detiene.
- Medido con los clientes oficiales (256 tokens, equipo frío): **28–29 tok/s con T=0 y 25–28 con T=1**.
  Ollama da 15,2 con el mismo modelo.
- No soporta: tools/function calling, imágenes ni presence/repeat penalty (los parámetros de Ollama para
  qwen3.5 incluyen `presence_penalty 1.5`; aquí se ignora). Solo escucha en 127.0.0.1 (`--host` para
  cambiarlo).

## Límite de hardware

DeepSeek‑V4.1‑Flash pesa **510 GB** (unos 552B parámetros, 16B activos en decode). No cabe en 24 GB de RAM
ni en el disco. Incluso V4‑Flash (284B) necesita Macs de 128 GB (el motor de referencia en Metal es
[antirez/ds4](https://github.com/antirez/ds4)). En este M4 los kernels se desarrollan y validan con formas
reales y pesos sintéticos. Para tokens/s de extremo a extremo hace falta un modelo que quepa en 24 GB.
