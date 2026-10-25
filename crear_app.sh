#!/bin/zsh
# Crea «Chat Qwen.app» (doble clic: arranca el modelo si hace falta y abre el chat).
# Guarda la ruta absoluta de chat.sh: si mueves la carpeta ds-metal, vuelve a ejecutarlo.
DIR="${0:A:h}"
osacompile -o "$DIR/Chat Qwen.app" -e "do shell script quoted form of \"$DIR/chat.sh\" & \" > /dev/null 2>&1 &\"" && echo "creada: $DIR/Chat Qwen.app"
