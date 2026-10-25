"""Vocabulario de borrador: los ids más frecuentes al tokenizar un corpus local de texto y código.

El lm_head de borrador solo puntúa estos ids (mucho más barato); si el token correcto no está, el
borrador falla y el modelo lo corrige, así que la salida no cambia, solo la aceptación.
Uso: python build_draft_vocab.py [raíz del corpus] [--max-mb 40]   (por defecto, la carpeta padre)
"""
import argparse
import collections
import os
from pathlib import Path

import numpy as np

import qwen_ollama
from mlx_lm.tokenizer_utils import load as load_tokenizer

EXTS = {".md", ".txt", ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".json", ".tex", ".sh"}
SKIP = {"node_modules", ".git", ".venv", "venv", "dist", "build", "__pycache__", ".next", "vendor"}
OUT = qwen_ollama.CACHE_DIR / "draft_vocab_counts.npy"


PROSE = {".md", ".txt", ".tex"}


def count(tok, root, exts, max_mb, max_file_kb):
    counts = collections.Counter()
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP and not d.startswith(".")]
        for f in filenames:
            p = Path(dirpath) / f
            if p.suffix not in exts:
                continue
            try:
                if p.stat().st_size > max_file_kb * 1024:
                    continue
                text = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            counts.update(tok.encode(text))
            total += len(text)
            if total > max_mb * 1e6:
                return counts, total
    return counts, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--max-mb", type=float, default=40)
    ap.add_argument("--max-file-kb", type=float, default=300)
    args = ap.parse_args()

    if not (qwen_ollama.CACHE_DIR / "tokenizer.json").exists():
        qwen_ollama.load()
    tok = load_tokenizer(qwen_ollama.CACHE_DIR)
    # Mezcla normalizada: todo (mucho código/JSON) + solo prosa. Solo con "todo", la prosa española
    # generada caía al 73-90 % de cobertura en el top-32k; con la mezcla, 93-98 % en el top-64k.
    arr = np.zeros(248320, dtype=np.float64)
    for name, exts in (("todo", EXTS), ("prosa", PROSE)):
        counts, total = count(tok, args.root, exts, args.max_mb, args.max_file_kb)
        n = sum(counts.values())
        for i, c in counts.items():
            if i < len(arr):
                arr[i] += c / n
        print(f"{name}: {total/1e6:.1f} MB de texto, {n} tokens, {len(counts)} ids distintos")
    np.save(OUT, arr)
    order = np.argsort(-arr)
    for k in (32768, 65536):
        print(f"top {k}: cubre {arr[order[:k]].sum()/arr.sum()*100:.2f}% del corpus mezclado")
    print("->", OUT)


if __name__ == "__main__":
    main()
