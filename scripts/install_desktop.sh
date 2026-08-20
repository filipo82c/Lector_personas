#!/bin/bash
# Script para instalar el Lector de Personas IA como aplicación nativa de escritorio en Linux

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=============================================================="
echo "   INSTALANDO LECTOR DE PERSONAS IA COMO APLICACIÓN DESKTOP"
echo "=============================================================="
echo ""

# 1. Crear carpeta de aplicaciones si no existe
mkdir -p ~/.local/share/applications
mkdir -p ~/.local/share/icons

# 2. Copiar ícono HD
cp "$DIR/assets/icon.png" ~/.local/share/icons/lector-personas.png

# 3. Crear el archivo .desktop para el Menú de Aplicaciones y Escritorio de Linux
CAT_FILE=~/.local/share/applications/lector-personas.desktop

cat <<EOF > "$CAT_FILE"
[Desktop Entry]
Version=1.0
Name=Lector de Personas IA
Comment=Sistema de Vigilancia Inteligente y Detección de Amenazas
Exec=/bin/bash "$DIR/iniciar_lector.sh"
Icon=$DIR/assets/icon.png
Terminal=false
Type=Application
Categories=Security;Utility;System;
Path=$DIR
EOF

chmod +x "$CAT_FILE"

# 4. Crear acceso directo en el Escritorio si existe
if [ -d "$HOME/Escritorio" ]; then
    cp "$CAT_FILE" "$HOME/Escritorio/"
    chmod +x "$HOME/Escritorio/lector-personas.desktop"
    gio trust "$HOME/Escritorio/lector-personas.desktop" 2>/dev/null || true
    echo "[OK] Acceso directo creado en: ~/Escritorio/lector-personas.desktop"
fi

if [ -d "$HOME/Desktop" ]; then
    cp "$CAT_FILE" "$HOME/Desktop/"
    chmod +x "$HOME/Desktop/lector-personas.desktop"
    gio trust "$HOME/Desktop/lector-personas.desktop" 2>/dev/null || true
    echo "[OK] Acceso directo creado en: ~/Desktop/lector-personas.desktop"
fi

echo "[OK] Aplicación agregada al Menú de Aplicaciones de Linux."
echo "=============================================================="
echo " ¡INSTALACIÓN COMPLETADA! Ya puedes buscar 'Lector de Personas IA'"
echo " en el menú de aplicaciones de tu computadora o abrirlo desde el escritorio."
echo "=============================================================="
