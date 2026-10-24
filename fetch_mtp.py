"""Descarga solo los tensores mtp.* de Qwen/Qwen3.5-9B con peticiones HTTP por rangos (~0.5 GB, no 18 GB)."""
import json
import struct
import sys
import urllib.request
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = "Qwen/Qwen3.5-9B"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
OUT = Path(__file__).parent / ".ollama_qwen" / "mtp.safetensors"


def get(url, start=None, end=None):
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    if OUT.exists():
        print("ya existe", OUT)
        return
    index = json.loads(get(BASE + "model.safetensors.index.json"))["weight_map"]
    shards = {}
    for k, f in index.items():
        if k.startswith("mtp."):
            shards.setdefault(f, []).append(k)
    out = {}
    for shard, keys in shards.items():
        url = BASE + shard
        n = struct.unpack("<Q", get(url, 0, 7))[0]
        header = json.loads(get(url, 8, 8 + n - 1))
        for k in keys:
            meta = header[k]
            a, b = meta["data_offsets"]
            raw = get(url, 8 + n + a, 8 + n + b - 1)
            assert meta["dtype"] == "BF16", meta
            u16 = np.frombuffer(raw, dtype=np.uint16).reshape(meta["shape"])
            out[k] = mx.array(u16).view(mx.bfloat16)
            print(f"{k:50s} {meta['shape']} {len(raw)/1e6:.1f} MB", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    mx.save_safetensors(str(OUT), out)
    print("guardado", OUT, sum(v.nbytes for v in out.values()) / 1e9, "GB")


if __name__ == "__main__":
    sys.exit(main())
