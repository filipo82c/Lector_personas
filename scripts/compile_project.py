import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST_DIR = os.path.join(BASE_DIR, "dist")
BUILD_DIR = os.path.join(BASE_DIR, "build")

def main():
    print("=================================================================")
    print("             COMPILADOR DEL PROYECTO LECTOR (PYINSTALLER)        ")
    print("=================================================================")

    # 1. Asegurar que PyInstaller este instalado
    try:
        import PyInstaller
        print("[OK] PyInstaller ya esta instalado en el entorno.")
    except ImportError:
        print("[!] PyInstaller no encontrado. Instalando via pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            print("[OK] PyInstaller instalado exitosamente.")
        except Exception as e:
            print(f"[ERROR] No se pudo instalar PyInstaller: {e}")
            sys.exit(1)

    # 2. Definir comando de PyInstaller
    # --onedir: Crea una carpeta autocomtenida (recomendado para IA por velocidad de carga y DLLs)
    # --console: Mantiene la consola abierta para ver los logs y alertas de la base de datos
    # --add-data: Incluye las carpetas iniciales modelos y db_rostros con sus archivos
    # --hidden-import: Fuerza la inclusion de librerias dinamicas importantes
    
    main_script = os.path.join(BASE_DIR, "src", "main.py")
    
    comando = [
        "pyinstaller",
        "--noconfirm",
        "--onedir",
        "--console",
        "--name=LectorPersonasIA",
        f"--add-data=modelos/*{os.pathsep}modelos/",
        f"--add-data=db_rostros/*{os.pathsep}db_rostros/",
        "--hidden-import=ultralytics",
        "--hidden-import=opencv-python",
        "--hidden-import=firebase-admin",
        "--hidden-import=sqlite3",
        main_script
    ]

    print("\n[INFO] Ejecutando compilacion (Esto puede tardar un momento)...")
    print("Comando:", " ".join(comando))
    
    try:
        subprocess.check_call(comando, cwd=BASE_DIR)
        print("\n=================================================================")
        print("[OK] ¡COMPILACION COMPLETADA CON EXITO!")
        print("=================================================================")
        print(f"El programa ejecutable se encuentra en:")
        print(f"  {os.path.join(DIST_DIR, 'LectorPersonasIA')}")
        print("\nPara ejecutar en la maquina de la empresa cliente:")
        print("1. Copia la carpeta 'LectorPersonasIA' completa.")
        print("2. Asegurate de que el archivo 'config.json' este en la raiz de esa carpeta para configurar camaras y bases de datos.")
        print("3. Ejecuta 'LectorPersonasIA.exe' haciendo doble clic.")
        print("=================================================================")
    except Exception as e:
        print(f"\n[ERROR] Error durante la compilacion con PyInstaller: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
