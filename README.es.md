# qwen-mtp-mac

**Qwen 3.5 9B unas 2× más rápido en Apple Silicon**. Usa la cabeza de predicción multi-token (MTP) del
propio modelo para la decodificación especulativa y añade unos cuantos kernels Metal sobre
[MLX](https://github.com/ml-explore/mlx). Incluye un servidor local compatible con Ollama y OpenAI y un chat
mínimo.

[Read in English](README.md)

En un **MacBook Air M4 de 24 GB** (sin ventilador), con los mismos pesos NVFP4:

| | tokens/s |
|---|---|
| Ollama `qwen3.5:9b-mlx` (greedy, 256 tokens) | 15,2 |
| **servidor qwen-mtp-mac** (greedy / temperatura 1, 256 tokens, clientes oficiales) | **31,6 / 32,3** |

Banco intercalado (`bench_ab.py`, 2 rondas × 4 prompts × 128 tokens). Las configuraciones se alternan
prompt a prompt para que el calor afecte a todas por igual:

| Configuración | tok/s | vs base | tokens/ciclo |
|---|---|---|---|
| mlx-lm `stream_generate` (`lm_head` a 6 bits) | 19,1 | 1,00× | 1 |
| MTP, 2 borradores por ciclo (`lm_head` a 6 bits) | 28,6 | 1,50× | 2,14 |
| **MTP, 2 borradores + `lm_head` en dos fases (por defecto)** | **31,4** | **1,65×** | 2,14 |

Con el Air ya caliente (3 rondas): base 16,9 y configuración por defecto 27,1 tok/s (1,60×).

## Instalación

Hace falta un Mac con Apple Silicon, Python 3.12 (con `uv`), [Ollama](https://ollama.com) con
`ollama pull qwen3.5:9b-mlx` (se leen sus pesos sin copiarlos; Ollama no tiene que estar en marcha) y unos
9 GB de RAM libres.

```bash
git clone https://github.com/aulafy/qwen-mtp-mac && cd qwen-mtp-mac
ollama pull qwen3.5:9b-mlx
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python fetch_mtp.py      # solo la cabeza MTP (~0,5 GB, por rangos HTTP)
.venv/bin/python server.py         # http://127.0.0.1:11435
```

Opcional: `.venv/bin/python build_draft_vocab.py ~/Documentos` cuenta las frecuencias de tokens en tus
textos para abaratar los borradores. Solo guarda recuentos, en `.ollama_qwen/`, fuera de git.

## Chat

`./chat.sh` arranca el servidor si hace falta y abre `http://127.0.0.1:11435/chat`. `./crear_app.sh` crea
`Chat Qwen.app` para el Dock; si mueves la carpeta, vuelve a ejecutarlo.

- **Contexto** (arriba): de 4K a 64K tokens, con un medidor de lo usado. Si se llena, se olvidan los
  mensajes más antiguos (las instrucciones y la memoria se conservan) y la respuesta lo indica.
- **Memoria**: notas que se añaden a todos los chats. Se guardan en `datos/memoria.md` y se pueden editar a
  mano. El botón de la bombilla con + resume el chat actual y lo añade.
- **Conversaciones**: se guardan solas en `datos/chats/*.json`.
- **Archivos**: texto, código, CSV, JSON, PDF con texto y DOCX (hasta 25 MB). El M4 los lee a ~160 tok/s:
  un archivo de 12K tokens tarda ~75 s la primera vez, y las preguntas siguientes del mismo chat reutilizan
  lo leído.
- **Ajustes por chat**: instrucciones, temperatura, razonar y usar o no la memoria. Se puede detener una
  respuesta a medias.
- **RAM**: el servidor se cierra solo tras 30 min sin uso (`QWEN_MTP_IDLE`) o con «Apagar modelo».
- Solo responde a páginas de 127.0.0.1/localhost: una web cualquiera no puede leer tus chats ni usar el
  modelo.

## Servidor (APIs de Ollama y OpenAI)

- **Ollama**: `GET /`, `/api/version`, `/api/tags`, `/api/ps`; `POST /api/chat`, `/api/generate`,
  `/api/show`. Opciones: `temperature`, `top_k`, `top_p`, `min_p`, `num_predict`, `num_ctx`, `stop`,
  `seed` y `think` (devuelve `message.thinking` aparte).
- **OpenAI**: `GET /v1/models`; `POST /v1/chat/completions` y `/v1/completions`, con streaming SSE y
  `stream_options.include_usage`. El razonamiento sale en `reasoning_content` con
  `chat_template_kwargs: {"enable_thinking": true}`.
- En cualquier app: URL de Ollama `http://127.0.0.1:11435` o base de OpenAI `http://127.0.0.1:11435/v1`
  (cualquier api key). `--port 11434` lo pone en el puerto de Ollama, con Ollama parado.
- Por defecto usa los parámetros de Ollama para qwen3.5 (T=1, top_k 20, top_p 0,95). `--greedy` pone T=0
  por defecto y `--think` activa el razonamiento si la petición no lo indica.
- **Caché entre turnos**: guarda el estado justo antes del prompt de generación y en el turno siguiente
  solo procesa lo nuevo.
- Un único hilo es dueño del modelo (los streams de MLX son por hilo). Las peticiones esperan turno y, si
  el cliente corta, la generación se detiene.

## Cómo funciona y qué se midió

Generar con un modelo de 9B a batch 1 está limitado por memoria: cada token lee ~4,5 GB de pesos y el
techo de lectura medido del M4 es ~99 GB/s (`bw_probe.py`). Hay dos formas de ganar: leer menos por token o
sacar más de un token por cada lectura de pesos.

**Lo que ganó** (en orden de llegada):

1. **`lm_head` cuantizado.** Ollama lo deja en bf16 y lee 2 GB por token. A 8 bits: 15,2 → 18,3 tok/s,
   coincidiendo en el 98,7 % de los top‑1 (KL 0,0002).
2. **Decodificación especulativa con la cabeza MTP nativa** (`mtp.py`, `fetch_mtp.py`). Ollama y mlx-lm la
   descartan. La cabeza propone tokens y el modelo los verifica de una pasada.
3. **`gdn_kernel.py`**: 24 de las 32 capas son GatedDeltaNet con estado recurrente. Para deshacer
   borradores fallidos, este dispatch único guarda el estado tras cada token y es idéntico bit a bit al de
   mlx-lm. Partir la llamada en dos se comía la ganancia (+25 ms por ciclo).
4. **2 borradores por ciclo**, con el **GEMV NVFP4 multi-fila** (`qmv_kernel.py`). El de MLX crece con M
   (3 filas +16 %, 4 filas +53 %); el nuestro, con productos `float4`, se queda plano: 294 frente a 339 µs y
   305 frente a 447. La primera versión, con cadenas de FMA dependientes, era más lenta que MLX (505 µs).
5. **Borrador barato**: el `lm_head` de los borradores solo puntúa los 64k ids más frecuentes de un corpus
   local, mezcla normalizada de código y prosa. Cubre el 93–98 % de lo que genera el modelo; solo con
   código, el top‑32k cubría el 73–90 %.
6. **Command buffers de MLX** (`tuning.py`). Con `MLX_MAX_OPS_PER_BUFFER=10` la GPU empieza antes: la
   pasada de 1 token baja de 56 a 52 ms y la de 3 de 70 a 62. Con el valor por defecto quedaba oculta la
   ganancia del punto 4. Hay que importar `tuning` antes que `mlx`.
7. **`lm_head` en dos fases** (`lm_head2.py`). Un grueso NVFP4 puntúa todo el vocabulario y se recalculan con
   los pesos bf16 originales solo los 64 mejores (13,2 → 7,4 ms con 3 filas). El top‑1 del bf16 cae siempre
   dentro del top‑16 del grueso, así que la salida es la del `lm_head` original: da los mismos 200 tokens
   que el bf16 completo en el mismo camino MTP. Para muestrear con top_k > 32 usa 4·top_k candidatos (256
   si top_k = 0). Cuesta ~1,8 GB más de RAM.

**Lo que no ganó**, medido igual:

- **GatedDeltaNet fusionada** (`gdn_fused.py`, desactivada). Reduce el tramo central de ~20 operaciones a 3
  y es idéntica bit a bit a mlx-lm: reproduce sus redondeos bf16 y sus fórmulas de sigmoid, softplus y
  rms_norm, comprobado sobre los 65.536 valores bf16. Aislada baja ese tramo de 4,0 a 2,9 ms, pero en la
  pasada completa no se nota.
- **Cabeza MTP a 8 bits y k adaptativo** (`tau`). Aciertan algo más (hasta 2,35 tokens por ciclo) pero
  cuestan más cálculo: empatan en frío y pierden con el Air caliente.
- **Borrador con los primeros 64k ids** (en vez de los más frecuentes): baja la aceptación y no gana.
- **3 borradores fijos**: el tercero acierta solo el 16–40 % de las veces.
- El experimento inicial, el MoE de DeepSeek‑V4.1 en Metal, empata con MLX (ver
  [`experiments/deepseek_moe`](experiments/deepseek_moe/README.md)).

`lab_gdn.py` separa el coste de las proyecciones (~10 ms, lectura de pesos) del tramo central de las
GatedDeltaNet (~3–4 ms). `bench_ab.py` admite configuraciones como `k2h`, `k2h8`, `k2ha` y `k3ha` (letras:
h `lm_head` en dos fases, 8 MTP a 8 bits, a k adaptativo, f borrador con vocabulario completo, g GDN
fusionada).

## Tests y bancos

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest -q tests      # exactitud de los kernels, sin el modelo
.venv/bin/python bench_ab.py             # banco intercalado
.venv/bin/python test_server.py          # servidor con los clientes oficiales openai/ollama
```

## Limitaciones

- Solo Qwen3.5-9B, leído del `qwen3.5:9b-mlx` de Ollama (NVFP4). En Hugging Face todavía no hay un
  checkpoint MLX en NVFP4; cargar otros checkpoints es el siguiente paso.
- Batch 1. Sin llamadas a herramientas, imágenes ni penalizaciones de repetición o presencia.
- En greedy, la salida sigue al greedy normal token a token hasta que un casi-empate en bf16 la desvía
  (verificar varios tokens a la vez cambia el redondeo). Sigue siendo texto del propio modelo.
- La velocidad depende de la temperatura del equipo: el Air baja de frecuencia en pocos minutos.
- Versiones fijadas (`mlx==0.31.2`, `mlx-lm==0.31.3`), porque se sustituyen métodos internos de mlx-lm.

## Créditos y licencia

MIT © 2026 Aulafy. Partes adaptadas de [mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT, Apple). El
enfoque de la decodificación especulativa sigue el de
[ml-explore/mlx-lm#990](https://github.com/ml-explore/mlx-lm/pull/990). Los pesos de Qwen3.5 son del equipo
Qwen (Apache 2.0) y no se incluyen. Ver [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Sin relación con
Alibaba/Qwen, Apple ni Ollama.
