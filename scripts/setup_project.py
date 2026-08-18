import os
import sys
import requests

# Configurar carpetas del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "modelos")
DB_DIR = os.path.join(BASE_DIR, "db_rostros")
ALERT_DIR = os.path.join(BASE_DIR, "alertas")
SRC_DIR = os.path.join(BASE_DIR, "src")

# URLs de los modelos a descargar
URLS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "threat_detection.pt": "https://huggingface.co/Subh775/Threat-Detection-YOLOv8n/resolve/main/weights/best.pt"
}

def descargar_con_progreso(url, path):
    """Descarga un archivo con una barra de progreso interactiva usando requests."""
    # Usar un user-agent para evitar que github o huggingface rechacen la petición
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    descargado = 0
    chunk_size = 1024 * 16 # 16KB chunks
    
    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                descargado += len(chunk)
                if total_size > 0:
                    porcentaje = min(100, int(descargado * 100 / total_size))
                    barra_len = 30
                    lleno_len = int(barra_len * porcentaje / 100)
                    barra = "=" * lleno_len + "-" * (barra_len - lleno_len)
                    sys.stdout.write(f"\r Descargando: [{barra}] {porcentaje}% ({descargado // 1024} KB / {total_size // 1024} KB)")
                    sys.stdout.flush()
    sys.stdout.write("\n")

def main():
    print("=== Inicializando Configuracion del Lector de Personas ===")
    
    # 1. Crear directorios si no existen
    for folder in [MODEL_DIR, DB_DIR, ALERT_DIR, SRC_DIR]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Creado directorio: {os.path.relpath(folder, BASE_DIR)}")
        else:
            print(f"El directorio ya existe: {os.path.relpath(folder, BASE_DIR)}")

    # 2. Descargar archivos de modelos
    for filename, url in URLS.items():
        dest_path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024 * 100:
            print(f"[OK] El modelo {filename} ya esta descargado localmente.")
        else:
            print(f" Descargando {filename} desde {url}...")
            try:
                descargar_con_progreso(url, dest_path)
                print(f"[OK] Descargado exitosamente en: {os.path.relpath(dest_path, BASE_DIR)}")
            except Exception as e:
                print(f"[ERROR] Error al descargar {filename}: {e}")
                # Borrar archivo corrupto si existe
                if os.path.exists(dest_path):
                    os.remove(dest_path)

    # 3. Descargar YOLO11 base usando Ultralytics (inicialización)
    print("\n Inicializando e importando YOLO11 para descargar el modelo base...")
    try:
        from ultralytics import YOLO
        yolo_path = os.path.join(MODEL_DIR, "yolo11n.pt")
        if not os.path.exists(yolo_path):
            print("Descargando yolo11n.pt a la carpeta modelos...")
            # Descargará automáticamente y lo guardará en el destino
            model = YOLO("yolo11n.pt")
            # Mover de la raíz a la carpeta modelos
            if os.path.exists("yolo11n.pt"):
                os.rename("yolo11n.pt", yolo_path)
                print(f"[OK] YOLO11 base movido a: {os.path.relpath(yolo_path, BASE_DIR)}")
        else:
            print("[OK] El modelo yolo11n.pt ya esta disponible localmente.")
    except ImportError:
        print("[!] No se pudo importar ultralytics. ¿Ya activaste el entorno virtual e instalaste requirements.txt?")
    except Exception as e:
        print(f"[!] Error al inicializar YOLO11: {e}")

    print("\n=== Configuracion Inicial Finalizada con Exito ===")

if __name__ == "__main__":
    main()
