"""Carga en mlx-lm el Qwen3.5 que ya tiene Ollama (qwen3.5:9b-mlx, NVFP4), sin descargar nada.

Los blobs de Ollama son safetensors de un tensor cada uno; los JSON (config, tokenizer) también
están en el manifiesto. Se enlazan los JSON en un directorio y los pesos se cargan blob a blob.
"""
import json
import os
from pathlib import Path

import tuning  # noqa: F401  (antes que mlx)
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5
from mlx_lm.tokenizer_utils import load as load_tokenizer

OLLAMA = Path(os.path.expanduser("~/.ollama/models"))
CACHE_DIR = Path(__file__).parent / ".ollama_qwen"
QUANT = {"group_size": 16, "bits": 4, "mode": "nvfp4"}


def _manifest(tag):
    m = json.loads((OLLAMA / "manifests/registry.ollama.ai/library/qwen3.5" / tag).read_text())
    blob = lambda l: OLLAMA / "blobs" / l["digest"].replace(":", "-")
    jsons = {l["name"]: blob(l) for l in m["layers"] if l["mediaType"].endswith("image.json")}
    tensors = [blob(l) for l in m["layers"] if l["mediaType"].endswith("image.tensor")]
    return jsons, tensors


def load(tag="9b-mlx", lm_head_quant=None):
    """lm_head_quant: None (bf16 como Ollama), "2f" (lm_head en dos fases, lm_head2.py) o dict para
    nn.quantize, p. ej. {"bits": 8, "group_size": 64}."""
    jsons, tensors = _manifest(tag)
    CACHE_DIR.mkdir(exist_ok=True)
    for name, path in jsons.items():
        link = CACHE_DIR / name
        if not link.exists():
            link.symlink_to(path)

    weights = {}
    for path in tensors:
        for k, v in mx.load(str(path), format="safetensors").items():
            if k.startswith("vision_tower"):
                continue
            weights[k.replace(".weight.scale", ".scales")] = v

    config = json.loads(jsons["config.json"].read_text())
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(config))
    weights = model.sanitize(weights)
    nn.quantize(model, **QUANT, class_predicate=lambda p, _: f"{p}.scales" in weights)
    model.load_weights(list(weights.items()), strict=True)
    if lm_head_quant == "2f":
        from lm_head2 import TwoStageHead

        model.language_model.lm_head = TwoStageHead(model.language_model.lm_head.weight)
    elif lm_head_quant:
        nn.quantize(model, **lm_head_quant, class_predicate=lambda p, _: p.endswith("lm_head"))
    mx.eval(model.parameters())
    model.eval()
    return model, load_tokenizer(CACHE_DIR)
