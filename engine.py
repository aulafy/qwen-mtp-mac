"""Motor de generación para el servidor: Qwen3.5-9B (pesos de Ollama) + MTP k=2 + kernels propios.

- Streaming de tokens por ciclo especulativo.
- Greedy (temperature 0) o muestreo especulativo EXACTO: el borrador es determinista (argmax de la
  MTP), así que se acepta con probabilidad p(d) y, si se rechaza, se muestrea de p sin d. La
  distribución de salida es la del modelo con temperature/top_k/top_p/min_p.
- Caché de prompt entre peticiones: se guarda el estado justo antes del prompt de generación
  (`<|im_start|>assistant\n...`), que es lo que se repite en el turno siguiente de un chat. Las KV
  caches se recortan y el estado de las GatedDeltaNet se restaura desde la instantánea.
"""
import queue
import threading
import time
from dataclasses import dataclass, field

import tuning  # noqa: F401  (antes que mlx)
import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

import mtp as M
import qmv_kernel
import qwen_ollama

PREFILL_CHUNK = 2048


@dataclass
class Sampling:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    seed: int | None = None


class Engine:
    def __init__(self, k=2, lm_head_bits=6, draft_vocab=65536):
        qmv_kernel.install()
        self.model, self.tok = qwen_ollama.load(lm_head_quant={"bits": lm_head_bits, "group_size": 64})
        self.lm = self.model.language_model
        self.mtp = M.load_mtp(self.lm, {"bits": 4, "group_size": 16, "mode": "nvfp4"})
        self.eos = set(self.tok.eos_token_ids)
        freq = (qwen_ollama.CACHE_DIR / "draft_vocab_counts.npy").exists()
        self.head = M.load_draft_head(draft_vocab if freq else 248320, {"bits": 4, "group_size": 64},
                                      freq, self.eos)
        self.k = k
        self._fresh()

    # ---------- caché ----------
    def _fresh(self):
        self.cache = self.lm.make_cache()
        self.mcache = [KVCache() for _ in self.mtp.layers]
        self.lin = [c for c in self.cache if isinstance(c, ArraysCache)]
        self.kv = [c for c in self.cache if isinstance(c, KVCache)]
        self.snap = None  # (ids, estados GDN, last_h)

    def _restore_snapshot(self):
        ids, states, last_h = self.snap
        n = len(ids)
        for c in self.kv:
            c.trim(c.offset - n)
        for c in self.mcache:
            c.trim(c.offset - (n - 1))
        for c, (conv, ssm) in zip(self.lin, states):
            c[0], c[1] = conv, ssm
        return n, last_h

    def _take_snapshot(self, ids, last_h):
        self.snap = (list(ids), [(c[0], c[1]) for c in self.lin], last_h)

    # ---------- prefill ----------
    def _prefill(self, ids, last_h):
        """Procesa `ids` en trozos; llena la KV de la MTP con los pares (H[i], tok[i+1])."""
        logits = None
        for a in range(0, len(ids), PREFILL_CHUNK):
            x = mx.array(ids[a:a + PREFILL_CHUNK])[None]
            logits, h, _ = M.backbone(self.lm, x, self.cache, last_only=True)
            if last_h is None:
                hid, toks = h[:, :-1], x[:, 1:]
            else:
                hid, toks = mx.concatenate([last_h, h[:, :-1]], axis=1), x
            ml = M.mtp_step(self.lm, self.mtp, hid, toks, self.mcache, self.head)[0] if toks.shape[1] else None
            last_h = h[:, -1:]
            mx.eval(logits, last_h, *([ml] if ml is not None else []))
        return logits, last_h

    # ---------- muestreo ----------
    @staticmethod
    def _dist(logits, sp):
        """logits [R, V] -> (idx [R, K] o None, logprobs [R, K]) tras temperature/top_k/top_p/min_p."""
        lp = logits.astype(mx.float32) / sp.temperature
        idx = None
        if 0 < sp.top_k < lp.shape[-1]:
            idx = mx.argpartition(-lp, sp.top_k - 1, axis=-1)[:, : sp.top_k]
            lp = mx.take_along_axis(lp, idx, axis=-1)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        if sp.top_p < 1.0:
            order = mx.argsort(-lp, axis=-1)
            srt = mx.take_along_axis(lp, order, axis=-1)
            keep_srt = (mx.cumsum(mx.exp(srt), axis=-1) - mx.exp(srt)) < sp.top_p
            keep = mx.put_along_axis(mx.zeros(lp.shape, dtype=mx.bool_), order, keep_srt, axis=-1)
            lp = mx.where(keep, lp, -mx.inf)
        if sp.min_p > 0:
            lp = mx.where(lp >= mx.max(lp, axis=-1, keepdims=True) + mx.log(sp.min_p), lp, -mx.inf)
        return idx, lp - mx.logsumexp(lp, axis=-1, keepdims=True)

    def _sample_first(self, logits, sp):
        if sp.temperature <= 0:
            return mx.argmax(logits[:, -1], axis=-1)[:, None]
        idx, lp = self._dist(logits[:, -1], sp)
        j = mx.random.categorical(lp)
        return (j if idx is None else mx.take_along_axis(idx, j[:, None], axis=-1)[:, 0])[:, None]

    def _verify(self, logits, drafts, sp):
        """logits [1, k+1, V], drafts [1, k] -> nº de borradores aceptados y tokens emitidos."""
        k = drafts.shape[1]
        dv = drafts[0]
        if sp.temperature <= 0:
            a = mx.argmax(logits[0], axis=-1)
            mx.eval(a)
            av, dl = a.tolist(), dv.tolist()
            n = 0
            while n < k and av[n] == dl[n]:
                n += 1
            return n, av[: n + 1]
        idx, lp = self._dist(logits[0], sp)
        if idx is None:
            eq = mx.arange(lp.shape[-1])[None, :] == dv[:, None]
        else:
            eq = idx[:k] == dv[:, None]
        p_d = (mx.exp(lp[:k]) * eq).sum(axis=-1)
        resid = mx.where(eq, -mx.inf, lp[:k])
        j_res = mx.random.categorical(resid)
        j_bonus = mx.random.categorical(lp[k:])
        if idx is not None:
            j_res = mx.take_along_axis(idx[:k], j_res[:, None], axis=-1)[:, 0]
            j_bonus = mx.take_along_axis(idx[k:], j_bonus[:, None], axis=-1)[:, 0]
        u = mx.random.uniform(shape=(k,))
        mx.eval(p_d, j_res, j_bonus, u)
        pd, ur, dl = p_d.tolist(), u.tolist(), dv.tolist()
        out = []
        for i in range(k):
            if ur[i] < pd[i]:
                out.append(dl[i])
            else:
                out.append(j_res.tolist()[i])
                return i, out
        out.append(j_bonus.item())
        return k, out

    # ---------- generación ----------
    def generate(self, ids, prefix_len, max_tokens, sp: Sampling, stats: dict):
        """Genera tokens (iterador) para el prompt `ids`. `prefix_len`: tokens del prompt sin el prompt
        de generación (punto de la instantánea reutilizable). Solo desde el hilo de Worker."""
        t0 = time.perf_counter()
        stats["t_start"] = t0
        if sp.seed is not None:
            mx.random.seed(sp.seed)
        reused, last_h = 0, None
        prefix_len = min(prefix_len, len(ids) - 1)  # el prompt de generación nunca puede quedar vacío
        if (self.snap is not None and len(self.snap[0]) <= prefix_len
                and ids[: len(self.snap[0])] == self.snap[0]):
            reused, last_h = self._restore_snapshot()
        else:
            self._fresh()
        if prefix_len > reused:
            _, last_h = self._prefill(ids[reused:prefix_len], last_h)
            self._take_snapshot(ids[:prefix_len], last_h)
        logits, last_h = self._prefill(ids[prefix_len:], last_h)
        t = self._sample_first(logits, sp)
        drafts = M.draft_chain(self.lm, self.mtp, last_h, t, self.mcache, self.k, self.head)
        mx.eval(t, drafts)
        t1 = time.perf_counter()
        stats.update(prompt_tokens=len(ids), cached_tokens=reused, t_first=t1)

        k, n_out, cycles, accepted = self.k, 0, 0, 0
        first = t.item()
        n_out += 1
        yield first
        done = first in self.eos or n_out >= max_tokens
        while not done:
            for c in self.lin:
                c.snap = True
            logits, h, _ = M.backbone(self.lm, mx.concatenate([t, drafts], axis=1), self.cache)
            for c in self.lin:
                c.snap = False
            n_acc, emitted = self._verify(logits, drafts, sp)
            cycles += 1
            accepted += n_acc
            keep = n_acc + 1
            if keep < k + 1:
                for c in self.lin:
                    M.restore(c, keep)
                for c in self.kv:
                    c.trim(k + 1 - keep)
            for c in self.mcache:
                c.trim(k - 1)
            seq = mx.array([emitted])
            t = seq[:, -1:]
            # pares (H[pos], token siguiente) de lo aceptado: los tokens son los emitidos
            drafts = M.draft_chain(self.lm, self.mtp, h[:, :keep], seq, self.mcache, k, self.head)
            for tok in emitted:
                n_out += 1
                yield tok
                if tok in self.eos or n_out >= max_tokens:
                    done = True
                    break
        stats.update(cycles=cycles, accepted=accepted, k=k)


@dataclass
class Job:
    args: tuple
    out: queue.Queue = field(default_factory=queue.Queue)
    cancel: threading.Event = field(default_factory=threading.Event)
    stats: dict = field(default_factory=dict)
    error: Exception | None = None
    done: bool = False

    def tokens(self):
        """Tokens a medida que salen; lanza el error del worker si lo hubo. Se puede volver a
        llamar para vaciar la cola tras cancelar (no se bloquea si ya terminó)."""
        while not self.done:
            t = self.out.get()
            if t is None:
                self.done = True
                break
            yield t
        if self.error:
            raise self.error


class Worker:
    """Un solo hilo dueño del modelo: los streams de MLX son por hilo y la GPU es una."""

    def __init__(self, **engine_kwargs):
        self.jobs = queue.Queue()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._loop, args=(engine_kwargs,), daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._init_error:
            raise self._init_error

    def _loop(self, kw):
        self._init_error = None
        try:
            self.engine = Engine(**kw)
        except Exception as e:  # noqa: BLE001 - se relanza en el hilo principal
            self._init_error = e
            self._ready.set()
            return
        self._ready.set()
        while True:
            job = self.jobs.get()
            try:
                for t in self.engine.generate(*job.args, job.stats):
                    if job.cancel.is_set():
                        break
                    job.out.put(t)
            except Exception as e:  # noqa: BLE001
                job.error = e
            finally:
                job.out.put(None)

    def submit(self, ids, prefix_len, max_tokens, sp):
        job = Job((ids, prefix_len, max_tokens, sp))
        self.jobs.put(job)
        return job
