# qwen-mtp-mac

**Qwen 3.5 9B twice as fast on an ordinary Mac, with the same output quality, 100 % local.**

[Leer en español](README.es.md)

## Goal

Run a modern AI model, **Qwen 3.5 9B**, on an everyday Mac (a fanless **MacBook Air M4 with 24 GB**) as fast
as possible:

- **no quality loss**: the text must be what the model itself would produce;
- **100 % local and private**: no internet, nothing leaves the machine;
- **easy to use**: a double-click chat, or a drop-in replacement for Ollama in any app.

## What we achieved

| Same Mac, same model, same weights | Speed |
|---|---|
| Ollama (`qwen3.5:9b-mlx`) | 15.2 tokens/s |
| **qwen-mtp-mac** | **31.6 tokens/s** (temperature 0) · **32.3** (temperature 1) |

- **~2× faster than Ollama** with the same weights.
- **Same quality.** Every proposed token is checked by the model before it is accepted, and the output layer
  (`lm_head`) picks the same token as the original uncompressed one.
- **A local chat** with memory, saved conversations, file reading (PDF, DOCX, code…) and a context-size
  selector.
- **Ollama- and OpenAI-compatible**: any app that talks to Ollama or the OpenAI API can use it by changing
  the address.

## Step-by-step install

You need an Apple Silicon Mac (tested on an M4; M1/M2 lack native bf16 on the GPU and will be slower),
~9 GB of free RAM and ~10 GB of disk.

1. **Download the model with Ollama.** Install [Ollama](https://ollama.com) and run:

   ```bash
   ollama pull qwen3.5:9b-mlx
   ```

   Ollama is only needed for this download (~9 GB). Afterwards you can quit or uninstall it, as long as you
   **keep the `~/.ollama/models` folder**, where the model files live.

2. **Install `uv`**, the Python manager, if you don't have it:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Get this project and set up its environment:**

   ```bash
   git clone https://github.com/aulafy/qwen-mtp-mac && cd qwen-mtp-mac
   uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
   ```

4. **Download the MTP head** (~0.5 GB), the part of the model that guesses the next tokens. Ollama does not
   include it:

   ```bash
   .venv/bin/python fetch_mtp.py
   ```

5. **Open the chat:**

   ```bash
   ./chat.sh         # starts the model and opens the chat in your browser
   ./crear_app.sh    # optional: builds "Chat Qwen.app" to open it from the Dock
   ```

6. **Optional: cheaper drafts.** Count the most frequent tokens in your own texts for a little more speed:

   ```bash
   .venv/bin/python build_draft_vocab.py ~/Documents
   ```

   Only token counts are stored, in `.ollama_qwen/`, which is not tracked by git.

## How to use it

### Chat

`./chat.sh` (or `Chat Qwen.app`) opens `http://127.0.0.1:11435/chat` (UI in Spanish):

- **Context**: choose 4K–64K tokens. A meter shows usage; when it fills up, the oldest messages are dropped
  and the answer says so.
- **Memory**: notes added to every chat (stored in `datos/memoria.md`). One button summarizes the current
  chat into memory.
- **Conversations** are saved in `datos/chats/`.
- **Files**: text, code, CSV, JSON, text PDFs and DOCX. The first read runs at ~160 tokens/s (a 12K-token
  file takes ~75 s); follow-up questions in the same chat do not read it again.
- **Per chat**: custom instructions, temperature, "reasoning" and memory on/off.
- **RAM**: the model unloads after 30 idle minutes (`QWEN_MTP_IDLE`) or with the "Apagar modelo" button.

### From other apps (Ollama and OpenAI APIs)

```bash
.venv/bin/python server.py   # http://127.0.0.1:11435
```

- **Ollama**: use `http://127.0.0.1:11435` as the server URL (`/api/chat`, `/api/generate`, `/api/tags`…,
  with `think` for reasoning).
- **OpenAI**: use `http://127.0.0.1:11435/v1` as the base URL with any API key (`/v1/chat/completions`,
  `/v1/completions`, `/v1/models`).
- `--port 11434` serves it on Ollama's port (with Ollama stopped). It only listens on your Mac (127.0.0.1)
  and rejects requests from foreign websites.

## How we got there, step by step

Text generation with a 9B model is limited by **memory**: every token reads ~4.5 GB of weights, and the M4
GPU reads at most ~99 GB/s (measured with `bw_probe.py`). There are only two ways to go faster: **read less
per token**, or **get several tokens out of each read**. Each step was measured before being kept:

| Step | What it does | Measured effect |
|---|---|---|
| 0. Starting point | Ollama, and mlx-lm with the same weights | 15.2 and 15.0–15.4 tok/s |
| 1. Quantized `lm_head` | Ollama leaves the output layer uncompressed: 2 GB extra per token | 15.2 → 18.3 tok/s |
| 2. Speculative decoding with the MTP head | The head guesses the next token, the model checks 2 at once (`mtp.py`) | 18.7 → 24.0 tok/s |
| 3. GatedDeltaNet rollback kernel | Saves the state after each token to undo misses in a single GPU pass (`gdn_kernel.py`) | Without it, step 2 lost 25 ms per cycle |
| 4. 2 drafts per cycle + multi-row GEMV | Kernel that reads each weight once for 3–4 tokens (`qmv_kernel.py`); drafts that only score frequent tokens | +9 % (19.6 → 21.3 tok/s, hot Mac) |
| 5. MLX command buffers | The GPU starts earlier (`tuning.py`) | +4.7 % (27.5 → 28.8 tok/s) |
| 6. Two-stage `lm_head` | Rough scoring of the whole vocabulary, exact scoring of the top 64 only (`lm_head2.py`) | +9.8 % (28.6 → 31.4 tok/s) |
| **Result** | Server with all of the above, official clients, 256 tokens | **31.6 tok/s vs 15.2 for Ollama** |

Steps 4–6 come from `bench_ab.py`, which alternates configurations prompt by prompt so that the Air's
thermal throttling (it slows down within minutes) affects all of them equally. Once the Air is hot, the
final configuration still gives 1.60× over mlx-lm (16.9 → 27.1 tok/s).

### Why quality is preserved

- **Greedy (temperature 0)**: a proposed token is accepted only if it is exactly the model's own choice.
- **With temperature**: exact speculative sampling. The draft is accepted with probability p(d), otherwise
  the token is sampled from the rest, so the distribution is the model's.
- **Two-stage `lm_head`**: the 64 candidates are recomputed with the original uncompressed weights. In our
  tests the chosen token always matched the original `lm_head` (the 6-bit version we used before matched
  99.2 % of the time).
- **One caveat**: checking several tokens at once changes bf16 rounding, so at a near-tie the output can
  take a different path than token-by-token generation. It is still the model's own text.

### What we tried and dropped

Measured the same way:

- **Fused GatedDeltaNet** (`gdn_fused.py`): bit-exact with mlx-lm and faster on its own (4.0 → 2.9 ms), but no
  gain in the full pass.
- **8-bit MTP head and adaptive number of drafts**: slightly more accurate drafts, but more compute. They tie
  when the Mac is cool and lose when it is hot.
- **3 fixed drafts**: the third one is accepted only 16–40 % of the time.
- **Drafting with the first 64k token ids** instead of the most frequent ones: lower acceptance.
- **The original experiment**: a DeepSeek‑V4.1 MoE kernel in Metal tied with MLX, and that model (510 GB)
  does not fit in a 24 GB Mac. See [`experiments/deepseek_moe`](experiments/deepseek_moe/README.md).

## Tests and benchmarks

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest -q tests      # kernel exactness (no model needed)
.venv/bin/python bench_ab.py             # interleaved benchmark: mlx-lm vs MTP variants
.venv/bin/python test_server.py          # server against the official OpenAI and Ollama clients
```

`bench_ab.py --configs` takes letter combinations: `h` two-stage `lm_head`, `8` 8-bit MTP, `a` adaptive k,
`f` full-vocabulary draft head and `g` fused GatedDeltaNet. `lab_gdn.py` measures the GatedDeltaNet layers on
their own.

## Limitations

- Only Qwen3.5-9B, with the `qwen3.5:9b-mlx` weights downloaded by Ollama (NVFP4). The Hugging Face checkpoint
  `mlx-community/Qwen3.5-9B-MLX-4bit` uses another format (affine 4-bit) that the kernels do not accelerate
  yet.
- One conversation at a time. No tool calling, images or repetition penalties.
- With `top_k = 0`, sampling is restricted to the 256 best candidates.
- Versions are pinned (`mlx==0.31.2`, `mlx-lm==0.31.3`) because the project replaces internal methods of
  mlx-lm.
- Code comments and the chat UI are in Spanish.

## Credits and license

MIT © 2026 Aulafy. Parts are adapted from [mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT, Apple). The
speculative-decoding approach follows [ml-explore/mlx-lm#990](https://github.com/ml-explore/mlx-lm/pull/990).
Qwen3.5 weights are released by the Qwen team under Apache 2.0 and are not included. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Not affiliated with Alibaba/Qwen, Apple or Ollama.
