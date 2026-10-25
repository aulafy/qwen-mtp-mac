"""Datos de la interfaz de chat: conversaciones, memoria y texto de archivos subidos.

Todo queda en datos/ (fuera de git):
  chats/<id>.json  una conversación (mensajes, ajustes, archivos adjuntos ya convertidos a texto)
  memoria.md       notas que se incluyen en todos los chats (editable a mano)
"""
import html
import io
import json
import os
import re
import time
import zipfile
from pathlib import Path

DATA = Path(__file__).parent / "datos"
CHATS = DATA / "chats"
MEMORY = DATA / "memoria.md"
ID_RE = re.compile(r"^[a-f0-9]{8,40}$")
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_TEXT_CHARS = 400_000
TEXT_EXTS = {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".html", ".htm", ".css",
             ".js", ".ts", ".tsx", ".jsx", ".py", ".rb", ".go", ".rs", ".java", ".kt", ".c", ".h", ".cpp",
             ".hpp", ".cs", ".php", ".sh", ".zsh", ".sql", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".log",
             ".tex", ".srt", ".vtt", ".swift", ".r", ".m", ".lua", ".pl", ".env", ".conf"}


def _path(cid):
    if not ID_RE.match(cid or ""):
        raise ValueError("id de conversación inválido")
    return CHATS / f"{cid}.json"


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def list_chats():
    out = []
    for p in CHATS.glob("*.json"):
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"id": c.get("id", p.stem), "title": c.get("title") or "Sin título",
                    "updated": c.get("updated", p.stat().st_mtime)})
    return sorted(out, key=lambda c: c["updated"], reverse=True)


def get_chat(cid):
    p = _path(cid)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_chat(cid, chat):
    chat = {**chat, "id": cid, "updated": time.time()}
    _write(_path(cid), json.dumps(chat, ensure_ascii=False, indent=1))
    return chat


def delete_chat(cid):
    p = _path(cid)
    if p.exists():
        p.unlink()


def get_memory():
    return MEMORY.read_text(encoding="utf-8") if MEMORY.exists() else ""


def set_memory(text):
    _write(MEMORY, text or "")


def _docx_text(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab/>", "\t", xml)
    return html.unescape(re.sub(r"<[^>]+>", "", xml))


def _pdf_text(data):
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [(p.extract_text() or "").strip() for p in reader.pages]
    return "\n\n".join(f"[página {i + 1}]\n{t}" for i, t in enumerate(pages) if t)


def extract_text(name, data):
    """Texto de un archivo subido. Lanza ValueError si el formato no se puede leer."""
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"archivo demasiado grande ({len(data) / 1e6:.0f} MB, máx {MAX_FILE_BYTES // 2**20} MB)")
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        try:
            text = _pdf_text(data)
        except Exception as e:  # noqa: BLE001 - pypdf lanza muchos tipos distintos
            raise ValueError(f"no se pudo leer el PDF ({e})") from e
        if not text.strip():
            raise ValueError("el PDF no tiene texto extraíble (¿escaneado?)")
    elif ext == ".docx":
        try:
            text = _docx_text(data)
        except (zipfile.BadZipFile, KeyError) as e:
            raise ValueError(f"no se pudo leer el DOCX ({e})") from e
    elif ext in TEXT_EXTS or b"\x00" not in data[:4096]:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
    else:
        raise ValueError(f"formato no soportado: {ext or 'binario'} (texto, código, CSV, PDF o DOCX)")
    truncated = len(text) > MAX_TEXT_CHARS
    return text[:MAX_TEXT_CHARS], truncated
