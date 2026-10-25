"""Decodificación especulativa con la cabeza MTP nativa de Qwen3.5 (greedy), sobre mlx-lm 0.31.3.

Cada ciclo: la cabeza MTP propone 1 token (d); el modelo verifica [t, d] en UNA pasada (los pesos
se leen una vez para los 2 tokens). Si acierta salen 2 tokens (d + el siguiente del modelo); si
falla sale 1 y se deshace el estado: las capas de atención recortan 1 posición de la KV cache y las
GatedDeltaNet recuperan el estado (conv + recurrente) guardado tras el primer token.
La salida es la misma que el greedy normal (salvo empates numéricos).

Uso: python mtp.py [--k 2] [--lm-head 2f|6|8] [--mtp-quant nvfp4|8|none] [--max-tokens 256]
"""
import argparse
import gc
import time
from pathlib import Path

import tuning  # noqa: F401  (antes que mlx)
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import stream_generate
from mlx_lm.models import qwen3_5
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import compute_g, gated_delta_update

import qwen_ollama
from gdn_kernel import gated_delta_snapshot
import qmv_kernel

MTP_PATH = Path(__file__).parent / ".ollama_qwen" / "mtp.safetensors"


class MTPLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.self_attn = qwen3_5.Attention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mlp = qwen3_5.MLP(args.hidden_size, args.intermediate_size)

    def __call__(self, x, mask, cache):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class MTP(nn.Module):
    """Predice el token t+2 a partir del hidden (pre-norm) en t y el embedding del token t+1."""

    def __init__(self, args):
        super().__init__()
        H = args.hidden_size
        self.pre_fc_norm_hidden = nn.RMSNorm(H, eps=args.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(H, eps=args.rms_norm_eps)
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.layers = [MTPLayer(args)]
        self.norm = nn.RMSNorm(H, eps=args.rms_norm_eps)


def load_mtp(lm, quant=None):
    mtp = MTP(lm.args)
    # Pesos crudos de HF: las RMSNorm de Qwen3.5 guardan (w - 1)
    w = {k.removeprefix("mtp."): (v + 1.0 if v.ndim == 1 else v)
         for k, v in mx.load(str(MTP_PATH)).items()}
    mtp.load_weights(list(w.items()), strict=True)
    if quant:
        nn.quantize(mtp, **quant)
    mx.eval(mtp.parameters())
    return mtp


def _gdn_call(self, inputs, mask=None, cache=None):
    """GatedDeltaNet.__call__ de mlx-lm 0.31.3 + cache.snap: guarda el estado tras cada token
    para poder volver a cualquier prefijo (restore)."""
    B, S, _ = inputs.shape
    snap = cache is not None and getattr(cache, "snap", False) and S > 1

    qkv = self.in_proj_qkv(inputs)
    z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
    b = self.in_proj_b(inputs)
    a = self.in_proj_a(inputs)

    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)

    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    n_keep = self.conv_kernel_size - 1
    if cache is not None:
        if cache.lengths is not None:
            ends = mx.clip(cache.lengths, 0, S)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_out = nn.silu(self.conv1d(conv_input))

    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            [self.head_k_dim, self.head_k_dim, self.head_v_dim],
        )
    ]
    state = cache[1] if cache else None
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

    if snap:
        if state is None:
            state = mx.zeros((B, self.num_v_heads, self.head_v_dim, self.head_k_dim), dtype=mx.float32)
        g = compute_g(self.A_log, a, self.dt_bias)
        out, state, mids = gated_delta_snapshot(q, k, v, g, mx.sigmoid(b), state)
        cache.rollback = (conv_input, mids, n_keep)
    else:
        out, state = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
                                        use_kernel=not self.training)

    if cache is not None:
        cache[1] = state
        cache.advance(S)

    out = self.norm(out, z)
    return self.out_proj(out.reshape(B, S, -1))


qwen3_5.GatedDeltaNet.__call__ = _gdn_call


def restore(cache, keep):
    """Deja una GatedDeltaNet como si solo hubiera procesado los `keep` primeros tokens del último paso."""
    conv_input, mids, n_keep = cache.rollback
    cache[0] = mx.contiguous(conv_input[:, keep:keep + n_keep, :])
    cache[1] = mids[keep - 1]


def backbone(lm, tokens, cache, last_only=False):
    """Como TextModel.__call__ pero devuelve también el hidden pre-norm y post-norm (entrada de la MTP).
    last_only: logits solo de la última posición (prefill: evita [S, 248320] logits)."""
    m = lm.model
    h = m.embed_tokens(tokens)
    fa_mask = create_attention_mask(h, cache[m.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[m.ssm_idx])
    for layer, c in zip(m.layers, cache):
        h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
    hn = m.norm(h)
    return lm.lm_head(hn[:, -1:] if last_only else hn), h, hn


class DraftHead:
    """lm_head barato solo para proponer. Con `ids` puntúa solo ese subconjunto del vocabulario."""

    def __init__(self, proj, ids=None):
        self.proj, self.ids = proj, ids

    def __call__(self, x):
        return self.proj(x)

    def token(self, logits):
        i = mx.argmax(logits, axis=-1)
        return i if self.ids is None else self.ids[i]


def load_draft_head(vocab, quant, freq=False, extra_ids=()):
    """`vocab` primeros ids o, con freq=True, los `vocab` más frecuentes del corpus local
    (build_draft_vocab.py), con menos bits. No cambia la salida (la verifica el lm_head completo),
    solo la aceptación."""
    jsons, tensors = qwen_ollama._manifest("9b-mlx")
    for path in tensors:
        w = mx.load(str(path), format="safetensors")
        if "language_model.lm_head.weight" in w:
            W = w["language_model.lm_head.weight"]
            break
    ids = None
    if freq:
        counts = np.load(qwen_ollama.CACHE_DIR / "draft_vocab_counts.npy")[: W.shape[0]]
        top = set(np.argsort(-counts)[:vocab].tolist()) | set(extra_ids)
        ids = mx.array(sorted(top), dtype=mx.uint32)
        W = mx.take(W, ids, axis=0)
    else:
        W = W[:vocab]
    head = nn.Linear(W.shape[1], W.shape[0], bias=False)
    head.weight = mx.contiguous(W)
    del w, W
    if quant:  # nn.quantize no sustituye el módulo raíz
        head = nn.QuantizedLinear.from_linear(head, **quant)
    mx.eval(head.parameters())
    gc.collect()  # suelta la copia bf16 de 2 GB del blob
    mx.clear_cache()
    return DraftHead(head, ids)


def mtp_step(lm, mtp, hidden, tokens, mcache, head=None):
    """Una pasada de la MTP sobre pares (hidden[i], tokens[i]); devuelve logits del último y su
    hidden pre-norm y post-norm (para encadenar el siguiente borrador)."""
    e = mtp.pre_fc_norm_embedding(lm.model.embed_tokens(tokens))
    x = mtp.fc(mx.concatenate([e, mtp.pre_fc_norm_hidden(hidden)], axis=-1))
    mask = create_attention_mask(x, mcache[0])
    for layer, c in zip(mtp.layers, mcache):
        x = layer(x, mask, c)
    x = x[:, -1:]
    xn = mtp.norm(x)
    return (head or lm.lm_head)(xn), x, xn


def draft_chain(lm, mtp, hidden, tokens, mcache, k, head=None, chain="pre"):
    """k borradores: el primero con los pares reales, los siguientes reutilizando el hidden de la propia MTP.
    Deja k-1 entradas especulativas en mcache (se recortan en el ciclo siguiente)."""
    pick = head.token if head is not None else (lambda l: mx.argmax(l, axis=-1))
    logits, x, xn = mtp_step(lm, mtp, hidden, tokens, mcache, head)
    drafts = [pick(logits)]
    for _ in range(k - 1):
        logits, x, xn = mtp_step(lm, mtp, x if chain == "pre" else xn, drafts[-1], mcache, head)
        drafts.append(pick(logits))
    return mx.concatenate(drafts, axis=1)


def generate_mtp(lm, mtp, prompt_ids, max_tokens, eos, head=None, k=1, bb_hidden="pre", chain="pre"):
    cache = lm.make_cache()
    mcache = [KVCache() for _ in mtp.layers]
    lin = [c for c in cache if isinstance(c, ArraysCache)]
    kv = [c for c in cache if isinstance(c, KVCache)]
    pick = (lambda h, hn: h) if bb_hidden == "pre" else (lambda h, hn: hn)

    p = mx.array(prompt_ids)[None]
    logits, h, hn = backbone(lm, p, cache)
    t = mx.argmax(logits[:, -1:], axis=-1)
    # la MTP ve los pares (H[i], token[i+1]) de todo el prompt y propone los primeros borradores
    drafts = draft_chain(lm, mtp, pick(h, hn), mx.concatenate([p[:, 1:], t], axis=1), mcache, k, head, chain)
    mx.eval(t, drafts)
    t0 = time.perf_counter()
    out = [t.item()]
    cycles = 0
    acc_pos = [0] * k  # aciertos por posición del borrador
    while len(out) < max_tokens and out[-1] not in eos:
        for c in lin:
            c.snap = True
        logits, h, hn = backbone(lm, mx.concatenate([t, drafts], axis=1), cache)
        for c in lin:
            c.snap = False
        a = mx.argmax(logits, axis=-1)  # [1, k+1]: predicción tras t, d1, ..., dk
        mx.eval(a)
        av, dv = a[0].tolist(), drafts[0].tolist()
        n_acc = 0
        while n_acc < k and av[n_acc] == dv[n_acc]:
            acc_pos[n_acc] += 1
            n_acc += 1
        cycles += 1
        keep = n_acc + 1  # t y los borradores aceptados
        out += av[:keep]  # borradores aceptados (== av) + el token corregido/extra del modelo
        if keep < k + 1:
            for c in lin:
                restore(c, keep)
            for c in kv:
                c.trim(k + 1 - keep)
        for c in mcache:
            c.trim(k - 1)
        t = a[:, keep - 1:keep]
        drafts = draft_chain(lm, mtp, pick(h, hn)[:, :keep], a[:, :keep], mcache, k, head, chain)
    dt = time.perf_counter() - t0
    for i, tok in enumerate(out):
        if tok in eos:
            out = out[: i + 1]
            break
    out = out[:max_tokens]
    stats = {"tps": (len(out) - 1) / dt, "tok_ciclo": (len(out) - 1) / max(cycles, 1),
             "acept": [n / max(cycles, 1) for n in acc_pos]}
    return out, stats


PROMPTS = [
    "Explica en detalle cómo funciona un motor de combustión interna de cuatro tiempos.",
    "Escribe una función en Python que valide un NIF español y explica cada paso.",
    "Resume las causas principales de la Revolución Francesa en una lista numerada.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-head", default="2f", help="2f (dos fases) | none (bf16) | 8 | 6 | 4 (bits)")
    ap.add_argument("--mtp-quant", default="nvfp4", help="none | nvfp4 | 8 | 4")
    ap.add_argument("--draft-vocab", type=int, default=65536, help="0 = usar el lm_head principal")
    ap.add_argument("--draft-bits", default="4")
    ap.add_argument("--draft-first-ids", action="store_true",
                    help="vocab de borrador = primeros ids (por defecto: los más frecuentes, build_draft_vocab.py)")
    ap.add_argument("--k", type=int, default=2, help="borradores por ciclo")
    ap.add_argument("--bb-hidden", default="pre", help="pre | post: hidden del modelo que ve la MTP")
    ap.add_argument("--chain", default="pre", help="pre | post: hidden de la MTP para encadenar")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--no-qmv", action="store_true", help="no usar el GEMV NVFP4 multi-fila propio")
    args = ap.parse_args()

    bits = lambda s: None if s == "none" else (
        {"bits": 4, "group_size": 16, "mode": "nvfp4"} if s == "nvfp4" else {"bits": int(s), "group_size": 64})
    if not args.no_qmv:
        qmv_kernel.install()
    model, tok = qwen_ollama.load(lm_head_quant="2f" if args.lm_head == "2f" else bits(args.lm_head))
    lm = model.language_model
    mtp = load_mtp(lm, bits(args.mtp_quant))
    eos = set(tok.eos_token_ids)
    freq = not args.draft_first_ids and (qwen_ollama.CACHE_DIR / "draft_vocab_counts.npy").exists()
    if args.draft_vocab and not freq and not args.draft_first_ids:
        print("sin draft_vocab_counts.npy (python build_draft_vocab.py): borrador con vocabulario completo")
        args.draft_vocab = 248320
    head = load_draft_head(args.draft_vocab, bits(args.draft_bits), freq, eos) if args.draft_vocab else None
    print(f"lm_head={args.lm_head} mtp={args.mtp_quant} draft={args.draft_vocab}/{args.draft_bits}b "
          f"k={args.k} bb={args.bb_hidden} chain={args.chain} mem {mx.get_active_memory()/1e9:.2f} GB")

    tot_base = tot_mtp = 0.0
    tot_cyc = 0.0
    for prompt in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, enable_thinking=False)
        if not args.no_baseline:
            ref, last = [], None
            for r in stream_generate(model, tok, ids, max_tokens=args.max_tokens):
                ref.append(r.token)
                last = r
            base_tps = last.generation_tps
        out, st = generate_mtp(lm, mtp, ids, args.max_tokens, eos, head, args.k, args.bb_hidden, args.chain)
        tps = st["tps"]
        tot_mtp += tps
        tot_cyc += st["tok_ciclo"]
        acc = " ".join(f"{x*100:4.1f}%" for x in st["acept"])
        if args.no_baseline:
            print(f"  mtp {tps:5.1f} tok/s  {st['tok_ciclo']:.2f} tok/ciclo  acept {acc}  "
                  f"({len(out)} tok)  {prompt[:40]}")
            continue
        same = next((i for i, (x, y) in enumerate(zip(out, ref)) if x != y), min(len(out), len(ref)))
        tot_base += base_tps
        print(f"  base {base_tps:5.1f} tok/s | mtp {tps:5.1f} tok/s ({tps/base_tps:.2f}x)  "
              f"{st['tok_ciclo']:.2f} tok/ciclo  acept {acc}  idénticos {same}/{len(ref)}  {prompt[:40]}")
    n = len(PROMPTS)
    print(f"media: mtp {tot_mtp/n:.1f} tok/s, {tot_cyc/n:.2f} tok/ciclo"
          + (f"  (base {tot_base/n:.1f} tok/s -> {tot_mtp/tot_base:.2f}x)" if tot_base else ""))
    print(tok.decode(out)[:300].replace("\n", " "))


if __name__ == "__main__":
    main()
