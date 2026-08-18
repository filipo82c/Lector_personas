import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "modelos")

def main():
    print("=================================================================")
    print("           ENTRENADOR AUTOMATICO DE IA - LECTOR                  ")
    print("=================================================================")

    # 1. Instalar roboflow si no esta
    try:
        from roboflow import Roboflow
        print("[OK] Libreria Roboflow disponible.")
    except ImportError:
        print("[!] Instalando roboflow...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])
        from roboflow import Roboflow

    # API Key de la cuenta Roboflow del usuario (extraida de su proyecto original)
    api_key_user = "Rmki1fTsv8sQP9T3mtuB"
    
    print("\n[1/3] Conectando a Roboflow y descargando dataset...")
    try:
        rf = Roboflow(api_key=api_key_user)
        project = rf.workspace("wd-pigj0").project("train-weapon")
        
        # Versión 3 es la del proyecto original, puedes cambiarla si subes más fotos a Roboflow
        version = project.version(3)
        
        print("[INFO] Descargando dataset en formato YOLOv8...")
        dataset_path = version.download("yolov8")
        
        yaml_path = os.path.join(dataset_path.location, "data.yaml")
        print(f"[OK] Dataset descargado y listo en: {dataset_path.location}")
        
    except Exception as e:
        print(f"[ERROR] No se pudo descargar el dataset de Roboflow: {e}")
        print("[CONSEJO] Asegurate de estar conectado a internet y que el API Key y proyecto sean validos.")
        sys.exit(1)

    # 2. Entrenar el modelo
    print("\n[2/3] Iniciando entrenamiento con Ultralytics YOLO...")
    try:
        from ultralytics import YOLO
        
        # Usamos yolov8n.pt como base preentrenada (para transfer learning)
        print("[INFO] Cargando modelo base preentrenado yolov8n.pt...")
        model = YOLO("yolov8n.pt")
        
        # Lanzar entrenamiento
        # device='cpu' usara los 16 hilos del Ryzen 7 5700G (muy rapido para sets pequeños)
        # Si tienes GPU NVIDIA/CUDA configurada puedes usar device=0.
        # Subimos las epocas a 50 para que aprenda mejor
        print("[INFO] Iniciando entrenamiento por 50 epocas...")
        model.train(
            data=yaml_path,
            epochs=50,
            imgsz=640,
            device="cpu",
            workers=4,
            project="entrenamientos_lector",
            name="armas_custom"
        )
        
        print("[OK] Entrenamiento completado con exito.")
        
    except Exception as e:
        print(f"[ERROR] Ocurrio un fallo durante el entrenamiento: {e}")
        sys.exit(1)

    # 3. Copiar el modelo entrenado
    # Las pesas de yolov8 se guardan en entrenamientos_lector/armas_custom/weights/best.pt
    best_weights_path = os.path.abspath(os.path.join(BASE_DIR, "entrenamientos_lector", "armas_custom", "weights", "best.pt"))
    dest_weights_path = os.path.join(MODEL_DIR, "threat_detection.pt")
    
    if os.path.exists(best_weights_path):
        import shutil
        try:
            shutil.copy(best_weights_path, dest_weights_path)
            print("\n=================================================================")
            print("[OK] ¡NUEVO MODELO ENTRENADO E INSTALADO CON EXITO!")
            print("=================================================================")
            print(f"El modelo fue copiado a: {dest_weights_path}")
            print("El sistema ahora usara tus pesas personalizadas en la proxima ejecucion.")
            print("=================================================================")
        except Exception as e:
            print(f"[!] No se pudo copiar el modelo a la carpeta final: {e}")
    else:
        print(f"[ERROR] No se encontraron las pesas resultantes en: {best_weights_path}")

if __name__ == "__main__":
    main()
