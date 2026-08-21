#!/bin/bash
# Script para conectar la carpeta a GitHub y hacer commit y push automáticamente

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=============================================================="
echo "      SUBIENDO CAMBIOS AUTOMÁTICAMENTE A GITHUB"
echo "=============================================================="
echo ""

# 1. Verificar si git está instalado
if ! command -v git &> /dev/null; then
    echo "[!] Git no se encuentra instalado en la consola actual."
    echo "Por favor instala git con: sudo apt install git"
    exit 1
fi

# 2. Inicializar .git si hace falta
if [ ! -d ".git" ]; then
    echo "[+] Inicializando repositorio git local..."
    git init
    git remote add origin https://github.com/filipo82c/Lector_personas.git
    git fetch origin main
    git reset --soft origin/main
fi

# 3. Configurar remoto e identidad git
git remote set-url origin https://github.com/filipo82c/Lector_personas.git

if [ -z "$(git config user.name)" ]; then
    git config user.name "filipo82c"
fi
if [ -z "$(git config user.email)" ]; then
    git config user.email "filipo82c@users.noreply.github.com"
fi

# 4. Agregar archivos, comitear y pushear
echo "[+] Agregando archivos modificados..."
git add .

echo "[+] Creando commit..."
git commit -m "Interfaz de escritorio nativa SOC, motor desacoplado y soporte multi-monitor"

# Garantizar que la rama local se llame main
git branch -M main

if command -v gh &> /dev/null; then
    gh auth setup-git 2>/dev/null || true
fi

echo "[+] Subiendo a GitHub (branch main)..."
git push -u origin main

echo ""
echo "=============================================================="
echo " ¡PROCESO COMPLETADO! Revisa tu repositorio en GitHub."
echo "=============================================================="
