"""lm_head en dos fases para Qwen3.5: grueso NVFP4 sobre todo el vocabulario y exacto (bf16 original)
solo para los K mejores candidatos.

Al verificar 3 tokens, el lm_head de 6 bits cuesta 13,2 ms en MLX (con 1 token, 8,4); el NVFP4 con el GEMV
multi-fila, 6,4 ms. El top-1 del lm_head bf16 cae dentro del top-16 del grueso en el 100 % de las
posiciones medidas y su top-20 dentro del top-64, así que se recalculan esos 64 con los pesos bf16:
el argmax (y el muestreo con top_k <= K/2) es el del lm_head original, no el de una copia cuantizada.
Devuelve logits de tamaño completo con -inf fuera de los candidatos, para no cambiar a quien los usa.
"""
import mlx.core as mx
import mlx.nn as nn


class TwoStageHead(nn.Module):
    def __init__(self, weight, k=64):
        super().__init__()
        lin = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        lin.weight = weight
        self.coarse = nn.QuantizedLinear.from_linear(lin, group_size=16, bits=4, mode="nvfp4")
        self.weight = weight  # bf16 [V, D]: solo se leen las filas de los candidatos
        self.k = k

    def __call__(self, h):
        c = self.coarse(h)
        idx = mx.argpartition(-c, self.k - 1, axis=-1)[..., : self.k]
        rows = mx.take(self.weight, idx, axis=0)  # [..., K, D]
        fine = (rows @ h[..., :, None])[..., 0]
        out = mx.full(c.shape, -mx.inf, dtype=fine.dtype)
        return mx.put_along_axis(out, idx, fine, axis=-1)


def k_for(sampling):
    """Candidatos según el muestreo: 64 para greedy o top_k <= 32; si no, 4·top_k (top_k = 0 -> 256)."""
    if sampling is None or sampling.temperature <= 0 or 0 < sampling.top_k <= 32:
        return 64
    return 256 if sampling.top_k <= 0 else min(4 * sampling.top_k, 2048)
