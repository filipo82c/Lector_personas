#!/bin/bash
echo "=============================================================="
echo "        INICIANDO SISTEMA DE SEGURIDAD INTELIGENTE"
echo "=============================================================="
echo ""

# Obtener el directorio del script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Verificar si el entorno virtual existe
if [ ! -f ".venv/bin/activate" ]; then
    echo "[ERROR] No se encontro el entorno virtual en .venv/"
    echo "Por favor, ejecuta primero scripts/setup_project.py"
    read -p "Presiona Enter para salir..."
    exit 1
fi

echo "[OK] Activando entorno virtual..."
source .venv/bin/activate

echo "[OK] Lanzando aplicacion principal de IA..."
python3 src/main.py

echo ""
echo "=============================================================="
echo "        EL SISTEMA HA FINALIZADO SU EJECUCION"
echo "=============================================================="
read -p "Presiona Enter para salir..."
