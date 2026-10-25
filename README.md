# qwen-mtp-mac

**Qwen 3.5 9B about 2× faster on Apple Silicon**, using the model's own multi-token-prediction (MTP) head
for speculative decoding plus a few custom Metal kernels on [MLX](https://github.com/ml-explore/mlx).
It ships an Ollama/OpenAI-compatible local server and a minimal chat app.

[Leer en español](README.es.md)

On a fanless **MacBook Air M4 (24 GB)**, with the same NVFP4 weights:

| | tokens/s |
|---|---|
| Ollama `qwen3.5:9b-mlx` (greedy, 256 tokens) | 15.2 |
| **qwen-mtp-mac server** (greedy / temperature 1, 256 tokens, official clients) | **31.6 / 32.3** |

Interleaved benchmark (`bench_ab.py`, 2 rounds × 4 prompts × 128 tokens, configurations alternated prompt by
prompt so that thermal throttling hits all of them equally):

| Configuration | tok/s | vs baseline | tokens per cycle |
|---|---|---|---|
| mlx-lm `stream_generate` (6-bit lm_head) | 19.1 | 1.00× | 1 |
| MTP, 2 drafts per cycle (6-bit lm_head) | 28.6 | 1.50× | 2.14 |
| **MTP, 2 drafts + two-stage lm_head (default)** | **31.4** | **1.65×** | 2.14 |

Once the Air is hot (3 rounds): baseline 16.9, default 27.1 tok/s (1.60×).

## How it works

Decoding a 9B model at batch 1 is memory-bound: each token reads ~4.5 GB of weights, and the measured
read ceiling of the M4 GPU is ~99 GB/s (`bw_probe.py`). So the gains come from reading less per token and
from producing more than one token per weight read:

- **Native MTP speculative decoding** (`mtp.py`, `engine.py`). Qwen3.5 checkpoints include an MTP head
  that Ollama and mlx-lm drop. It is downloaded on its own (~0.5 GB via HTTP range requests,
  `fetch_mtp.py`). Each cycle it proposes 2 draft tokens and the model verifies `[t, d1, d2]` in one pass.
  Greedy verification is exact; with temperature it uses exact speculative sampling (the draft is
  deterministic: accept with probability p(d), otherwise sample from p without d).
- **GatedDeltaNet rollback kernel** (`gdn_kernel.py`). 24 of the 32 layers are linear-attention
  GatedDeltaNets with a recurrent state. Rejected drafts must be undone, so this single Metal dispatch
  writes the state after every token; it is bit-exact with mlx-lm's kernel.
- **Multi-row NVFP4 GEMV** (`qmv_kernel.py`). MLX's quantized matmul gets slower with 3–4 rows (+16 % and
  +53 %). This one reads each weight once for all rows: 294 µs vs 339 (3 rows) and 305 vs 447 (4 rows) on a
  12288×4096 layer, with the same results.
- **Two-stage lm_head** (`lm_head2.py`). The 248k-token vocabulary head is scored with a coarse NVFP4 copy;
  only the top 64 candidates are recomputed with the original bf16 weights (7.4 ms instead of 13.2 for
  3 rows). In every measured position the bf16 top-1 was inside the coarse top-16, so the output matches the
  original bf16 head (a 6-bit head agreed 99.2 % of the time).
- **Cheap draft head**. The MTP drafts score only the 64k most frequent token ids of a local text corpus,
  4-bit (`build_draft_vocab.py`, optional; without it the full vocabulary is used).
- **MLX command buffers** (`tuning.py`). `MLX_MAX_OPS_PER_BUFFER=10` lets the GPU start earlier: a 1-token
  pass goes from 56 to 52 ms, a 3-token pass from 70 to 62 ms.

What did **not** help, measured the same way: a fused GatedDeltaNet kernel (`gdn_fused.py`, bit-exact
with mlx-lm, but no end-to-end gain), an 8-bit MTP head, adaptive draft length, and drafting with the first
64k token ids. The [Spanish README](README.es.md) has the full log with numbers.

## Requirements

- Apple Silicon Mac (developed and measured on an M4 with 24 GB) and macOS.
- Python 3.12 (`uv` recommended).
- [Ollama](https://ollama.com) with the MLX weights pulled: `ollama pull qwen3.5:9b-mlx`. The weights are read
  from Ollama's local store and nothing is copied; Ollama does not need to be running.
- ~9 GB of free RAM for the server, and ~0.5 GB of disk for the MTP head.

## Quick start

```bash
git clone https://github.com/aulafy/qwen-mtp-mac && cd qwen-mtp-mac
ollama pull qwen3.5:9b-mlx
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python fetch_mtp.py      # MTP head only (~0.5 GB)
.venv/bin/python server.py         # http://127.0.0.1:11435
```

Optional: `.venv/bin/python build_draft_vocab.py ~/Documents` counts token frequencies in your own texts
for the cheaper draft head. Only counts are stored, in `.ollama_qwen/`, outside git.

### Chat app

`./chat.sh` starts the server if needed and opens `http://127.0.0.1:11435/chat`. `./crear_app.sh` builds
`Chat Qwen.app` for the Dock. The chat stores conversations in `datos/chats/` and a global memory in
`datos/memoria.md`. It lets you pick the context size (4K–64K) and reads text, code, CSV, JSON, PDF and
DOCX files. The server exits after 30 idle minutes (`QWEN_MTP_IDLE`) to give the RAM back. The UI is in
Spanish.

### API

Point any client to `http://127.0.0.1:11435` (Ollama) or `http://127.0.0.1:11435/v1` (OpenAI, any key):

- Ollama: `GET /api/tags`, `/api/version`, `/api/ps`; `POST /api/chat`, `/api/generate`, `/api/show`.
  Options: `temperature`, `top_k`, `top_p`, `min_p`, `num_predict`, `num_ctx`, `stop`, `seed`, `think`.
- OpenAI: `GET /v1/models`; `POST /v1/chat/completions`, `/v1/completions`, streaming included.
- One generation at a time. The prompt cache is reused across chat turns. The server listens on 127.0.0.1
  only and rejects non-local browser origins.

## Benchmarks and tests

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest -q tests      # kernel exactness, no model needed
.venv/bin/python bench_ab.py             # interleaved baseline vs MTP configurations
.venv/bin/python test_server.py          # server against the official openai/ollama clients
```

## Limitations

- Only Qwen3.5-9B, read from Ollama's `qwen3.5:9b-mlx` (NVFP4). There is no NVFP4 MLX checkpoint on Hugging
  Face yet; loading other checkpoints is the next step.
- Batch size 1. No tool calling, images or presence/repeat penalties.
- With `top_k = 0`, sampling is restricted to the 256 best candidates of the two-stage head.
- Greedy output follows plain greedy decoding token by token until a bf16 near-tie (verifying several
  tokens at once changes the rounding). It is still the model's own output.
- Speeds depend on temperature and thermals; a fanless Air throttles within minutes.
- Versions are pinned (`mlx==0.31.2`, `mlx-lm==0.31.3`) because the project replaces internal methods of
  mlx-lm.
- Code comments and the chat UI are in Spanish.

## Credits and license

MIT © 2026 Aulafy. Parts are adapted from [mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT, Apple). The
speculative-decoding approach follows [ml-explore/mlx-lm#990](https://github.com/ml-explore/mlx-lm/pull/990).
Qwen3.5 weights are released by the Qwen team under Apache 2.0 and are not included. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Not affiliated with Alibaba/Qwen, Apple or Ollama.
