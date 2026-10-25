# qwen-mtp-mac

**Qwen 3.5 9B el doble de rápido en un Mac normal, sin perder calidad y 100 % en local.**

[Read in English](README.md)

## Objetivo

Que un modelo de IA moderno, **Qwen 3.5 9B**, funcione en un Mac corriente (un **MacBook Air M4 de 24 GB**,
sin ventilador) lo más rápido posible:

- **sin perder calidad**: el texto tiene que ser el que daría el propio modelo;
- **100 % local y privado**: sin internet y sin enviar nada fuera;
- **fácil de usar**: como chat con doble clic, o como sustituto de Ollama en cualquier app.

## Qué hemos conseguido

| Mismo Mac, mismo modelo y mismos pesos | Velocidad |
|---|---|
| Ollama (`qwen3.5:9b-mlx`) | 15,2 tokens/s |
| **qwen-mtp-mac** | **31,6 tokens/s** (temperatura 0) · **32,3** (temperatura 1) |

- **~2× más rápido que Ollama** con los mismos pesos.
- **Misma calidad.** El modelo comprueba cada token propuesto antes de darlo por bueno, y la última capa
  (`lm_head`) devuelve el mismo token que la original sin comprimir.
- **Un chat local** con memoria, conversaciones guardadas, archivos (PDF, DOCX, código…) y elección del
  contexto.
- **Compatible con Ollama y OpenAI**: cualquier app que hable con Ollama o con la API de OpenAI puede usarlo
  cambiando solo la dirección.

## Instalación paso a paso

Necesitas un Mac con Apple Silicon (probado en un M4; en M1/M2, sin bf16 nativo en la GPU, irá más
lento), unos 9 GB de RAM libres y ~10 GB de disco.

1. **Descarga el modelo con Ollama.** Instala [Ollama](https://ollama.com) y ejecuta:

   ```bash
   ollama pull qwen3.5:9b-mlx
   ```

   Ollama solo hace falta para esta descarga (~9 GB). Luego puedes cerrarlo o desinstalarlo, siempre que
   **no borres la carpeta `~/.ollama/models`**, donde quedan los archivos del modelo.

2. **Instala `uv`**, el gestor de Python, si no lo tienes:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Descarga este proyecto y prepara su entorno:**

   ```bash
   git clone https://github.com/aulafy/qwen-mtp-mac && cd qwen-mtp-mac
   uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
   ```

4. **Descarga la cabeza MTP** (~0,5 GB). Es la pieza del modelo que adivina los siguientes tokens; Ollama no
   la incluye:

   ```bash
   .venv/bin/python fetch_mtp.py
   ```

5. **Abre el chat:**

   ```bash
   ./chat.sh         # arranca el modelo y abre el chat en el navegador
   ./crear_app.sh    # opcional: crea «Chat Qwen.app» para abrirlo con doble clic desde el Dock
   ```

6. **Opcional: borradores más baratos.** Cuenta las palabras más frecuentes en tus propios textos para
   acelerar un poco más:

   ```bash
   .venv/bin/python build_draft_vocab.py ~/Documentos
   ```

   Solo se guardan recuentos de tokens, en `.ollama_qwen/`, que no se sube a git.

## Cómo usarlo

### Chat

`./chat.sh` (o `Chat Qwen.app`) abre `http://127.0.0.1:11435/chat`:

- **Contexto**: elige de 4K a 64K tokens. Un medidor muestra lo usado; si se llena, se olvidan los mensajes
  más antiguos y la respuesta lo indica.
- **Memoria**: notas que se incluyen en todos los chats (botón «Memoria», guardadas en `datos/memoria.md`).
  El botón de la bombilla con + resume el chat actual y lo añade.
- **Conversaciones**: se guardan solas en `datos/chats/`.
- **Archivos**: texto, código, CSV, JSON, PDF con texto y DOCX. La primera lectura va a ~160 tokens/s (un
  archivo de 12K tokens tarda ~75 s); las preguntas siguientes del mismo chat ya no lo vuelven a leer.
- **Por chat**: instrucciones propias, temperatura, «razonar» y usar o no la memoria.
- **RAM**: el modelo se apaga solo tras 30 min sin uso (`QWEN_MTP_IDLE`) o con «Apagar modelo».

### Desde otras apps (API de Ollama y OpenAI)

```bash
.venv/bin/python server.py   # http://127.0.0.1:11435
```

- **Ollama**: usa `http://127.0.0.1:11435` como dirección del servidor (`/api/chat`, `/api/generate`,
  `/api/tags`…, con `think` para el razonamiento).
- **OpenAI**: usa `http://127.0.0.1:11435/v1` como base, con cualquier API key (`/v1/chat/completions`,
  `/v1/completions`, `/v1/models`).
- `--port 11434` lo pone en el puerto de Ollama (con Ollama parado). Solo escucha en tu Mac (127.0.0.1) y
  rechaza peticiones de webs ajenas.

## Cómo lo hemos conseguido, paso a paso

Generar texto con un modelo de 9B está limitado por la **memoria**: por cada token hay que leer unos
4,5 GB de pesos, y la GPU del M4 lee como mucho ~99 GB/s (medido con `bw_probe.py`). Solo hay dos formas de
ir más rápido: **leer menos por token** o **sacar varios tokens por cada lectura**. Cada paso se midió antes
de darlo por bueno:

| Paso | Qué hace | Efecto medido |
|---|---|---|
| 0. Punto de partida | Ollama, y mlx-lm con los mismos pesos | 15,2 y 15,0–15,4 tok/s |
| 1. `lm_head` cuantizado | Ollama deja la última capa sin comprimir y lee 2 GB extra por token | 15,2 → 18,3 tok/s |
| 2. Decodificación especulativa con la cabeza MTP | La cabeza adivina el siguiente token y el modelo comprueba 2 de golpe (`mtp.py`) | 18,7 → 24,0 tok/s |
| 3. Kernel para deshacer en GatedDeltaNet | Guarda el estado tras cada token para deshacer los fallos en un solo paso de GPU (`gdn_kernel.py`) | Sin él, el paso 2 perdía 25 ms por ciclo |
| 4. 2 borradores por ciclo + GEMV multi-fila | Kernel que lee cada peso una vez para 3–4 tokens (`qmv_kernel.py`) y borrador que solo puntúa los tokens frecuentes | +9 % (19,6 → 21,3 tok/s, Mac caliente) |
| 5. Command buffers de MLX | La GPU empieza antes (`tuning.py`) | +4,7 % (27,5 → 28,8 tok/s) |
| 6. `lm_head` en dos fases | Puntuación aproximada de todo el vocabulario y exacta solo de los 64 mejores (`lm_head2.py`) | +9,8 % (28,6 → 31,4 tok/s) |
| **Resultado** | Servidor con todo lo anterior, clientes oficiales, 256 tokens | **31,6 tok/s frente a 15,2 de Ollama** |

Las cifras de los pasos 4–6 salen de `bench_ab.py`, que alterna las configuraciones prompt a prompt para
que el calor del Air (que baja de frecuencia en pocos minutos) afecte a todas por igual. Con el Air ya
caliente, la configuración final sigue dando 1,60× sobre mlx-lm (16,9 → 27,1 tok/s).

### Por qué no se pierde calidad

- **Greedy (temperatura 0)**: un token propuesto solo se acepta si es exactamente el que elegiría el modelo.
- **Con temperatura**: muestreo especulativo exacto. Se acepta el borrador con probabilidad p(d) y, si se
  rechaza, se muestrea del resto, así que la distribución es la del modelo.
- **`lm_head` en dos fases**: los 64 candidatos se recalculan con los pesos originales sin comprimir. En las
  pruebas, el token elegido coincidió siempre con el del `lm_head` original (la versión de 6 bits que
  usábamos antes coincidía en el 99,2 %).
- **Única salvedad**: comprobar varios tokens a la vez cambia el redondeo en bf16, y en un casi-empate la
  salida puede tomar otro camino que la generación token a token. Sigue siendo texto del propio modelo.

### Lo que probamos y no funcionó

Medido igual, y descartado:

- **GatedDeltaNet fusionada** (`gdn_fused.py`). Idéntica bit a bit a mlx-lm y más rápida aislada (4,0 →
  2,9 ms), pero sin ganancia en la pasada completa.
- **Cabeza MTP a 8 bits y número de borradores adaptativo.** Aciertan algo más, pero calculan más: empatan
  en frío y pierden con el Mac caliente.
- **3 borradores fijos.** El tercero acierta solo el 16–40 % de las veces.
- **Borrador con los primeros 64k tokens** en vez de los más frecuentes: baja la aceptación.
- **El experimento inicial**: el MoE de DeepSeek‑V4.1 en Metal empata con MLX, y ese modelo (510 GB) no
  cabe en un Mac de 24 GB. Ver [`experiments/deepseek_moe`](experiments/deepseek_moe/README.md).

## Tests y bancos

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest -q tests      # exactitud de los kernels (no necesita el modelo)
.venv/bin/python bench_ab.py             # banco intercalado: mlx-lm frente a las variantes MTP
.venv/bin/python test_server.py          # servidor con los clientes oficiales de OpenAI y Ollama
```

`bench_ab.py --configs` admite combinaciones de letras: `h` `lm_head` en dos fases, `8` MTP a 8 bits,
`a` k adaptativo, `f` borrador con vocabulario completo y `g` GatedDeltaNet fusionada. `lab_gdn.py` mide
por separado el coste de las capas GatedDeltaNet.

## Limitaciones

- Solo Qwen3.5-9B, con los pesos `qwen3.5:9b-mlx` descargados con Ollama (formato NVFP4). El checkpoint de
  Hugging Face `mlx-community/Qwen3.5-9B-MLX-4bit` usa otro formato (4 bits afín) que los kernels todavía no
  aceleran.
- Una conversación a la vez. Sin llamadas a herramientas, imágenes ni penalizaciones de repetición.
- Con `top_k = 0`, el muestreo se limita a los 256 mejores candidatos.
- Versiones fijadas (`mlx==0.31.2`, `mlx-lm==0.31.3`), porque se sustituyen métodos internos de mlx-lm.
- Comentarios del código e interfaz del chat en español.

## Créditos y licencia

MIT © 2026 Aulafy. Partes adaptadas de [mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT, Apple). El
enfoque de la decodificación especulativa sigue el de
[ml-explore/mlx-lm#990](https://github.com/ml-explore/mlx-lm/pull/990). Los pesos de Qwen3.5 son del equipo
Qwen (Apache 2.0) y no se incluyen. Ver [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Sin relación con
Alibaba/Qwen, Apple ni Ollama.
