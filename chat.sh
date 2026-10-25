#!/bin/zsh
# Arranca el modelo (si no está ya en marcha) y abre el chat en el navegador.
# Lo usa la app «Chat Qwen.app». El servidor se cierra solo tras DS_METAL_IDLE minutos sin uso
# (30 por defecto) para devolver los ~7,5 GB de RAM; el siguiente doble clic lo vuelve a cargar.
DIR="${0:A:h}"
PORT="${DS_METAL_PORT:-11435}"
URL="http://127.0.0.1:$PORT/chat"
LOG="$DIR/datos/servidor.log"
cd "$DIR" || exit 1
mkdir -p datos

notify() { osascript -e "display notification \"$1\" with title \"Chat Qwen\"" >/dev/null 2>&1 }

if ! curl -sf "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then
  notify "Cargando el modelo…"
  [ -f .ollama_qwen/mtp.safetensors ] || .venv/bin/python fetch_mtp.py >>"$LOG" 2>&1
  [ -f .ollama_qwen/draft_vocab_counts.npy ] || .venv/bin/python build_draft_vocab.py >>"$LOG" 2>&1
  nohup .venv/bin/python server.py --port "$PORT" --idle-exit "${DS_METAL_IDLE:-30}" >>"$LOG" 2>&1 &
  for _ in {1..240}; do
    curl -sf "http://127.0.0.1:$PORT/" >/dev/null 2>&1 && break
    sleep 0.5
  done
  if ! curl -sf "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then
    notify "No arrancó: mira datos/servidor.log"
    exit 1
  fi
fi
[ -n "$DS_METAL_NO_OPEN" ] || open "$URL"
