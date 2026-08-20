#!/bin/bash
# Obtener el directorio absoluto del proyecto
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Ejecutar usando el entorno virtual .venv directamente
if [ -f "$DIR/.venv/bin/python3" ]; then
    "$DIR/.venv/bin/python3" "$DIR/src/main.py" "$@"
elif [ -f "$DIR/.venv/bin/python" ]; then
    "$DIR/.venv/bin/python" "$DIR/src/main.py" "$@"
else
    python3 "$DIR/src/main.py" "$@"
fi
