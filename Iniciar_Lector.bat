@echo off
title Sistema de Seguridad - Lector Personas e IA
echo ==============================================================
echo         INICIANDO SISTEMA DE SEGURIDAD INTELIGENTE
echo ==============================================================
echo.
cd /d "%~dp0"

:: Verificar si el entorno virtual existe
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] No se encontro el entorno virtual en .venv\
    echo Por favor, ejecuta primero scripts/setup_project.py
    pause
    exit /b
)

echo [OK] Activando entorno virtual...
call .venv\Scripts\activate.bat

echo [OK] Lanzando aplicacion principal de IA...
python src/main.py

echo.
echo ==============================================================
echo        EL SISTEMA HA FINALIZADO SU EJECUCION
echo ==============================================================
pause
