import sys
import os
import time
import threading
import cv2

# Agregar src/ al PATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import database
import detector
import camera

def main():
    print("=================================================================")
    print("          LECTOR DE PERSONAS - SISTEMA DE IA MULTIHILO           ")
    print("=================================================================")
    print(f"Directorio de Trabajo: {config.BASE_DIR}")
    print(f"Dispositivos configurados: {len(config.CAMARAS)}")
    print("=================================================================\n")

    # 1. Inicializar Base de Datos (SQLite Local o Firebase Cloud)
    print("Inicializando Base de Datos...")
    db = database.obtener_base_datos()

    # 2. Inicializar Detector Central de IA (Carga YOLO11 y modelos faciales una sola vez)
    print("\nInicializando Motores de Inteligencia Artificial...")
    try:
        detector_ia = detector.SecurityDetector()
    except Exception as e:
        print(f"[ERROR CRITICO] No se pudieron cargar los modelos de IA: {e}")
        db.cerrar()
        sys.exit(1)

    print(f"\n[SISTEMA OK] Base de datos de rostros lista con {len(detector_ia.db_embeddings)} registros.")

    # 3. Lanzar Hilos de Camaras
    latest_frames = {}
    stop_event = threading.Event()
    workers = []

    print("\nLanzando hilos de procesamiento por camara...")
    for cam_info in config.CAMARAS:
        cam_id = cam_info["id"]
        source = cam_info["source"]
        
        worker = camera.CameraWorker(
            camera_id=cam_id,
            source=source,
            detector=detector_ia,
            db=db,
            latest_frames=latest_frames,
            stop_event=stop_event
        )
        worker.start()
        workers.append(worker)

    print("\n[INFO] Sistema en ejecucion.")
    if config.SISTEMA["mostrar_ventanas"]:
        print("[INFO] Presiona la tecla 'q' en cualquier ventana de video para salir.")
    else:
        print("[INFO] Ejecutando en modo headless (segundo plano). Presiona Ctrl+C en consola para salir.")

    # 4. Bucle Principal en el Hilo de Ejecucion Central (Main Thread)
    try:
        while not stop_event.is_set():
            if config.SISTEMA["mostrar_ventanas"]:
                # Copia rapida de las llaves para evitar errores de iteracion en multihilo
                cameras_active = list(latest_frames.keys())
                
                for cam_id in cameras_active:
                    frame = latest_frames.get(cam_id)
                    if frame is not None:
                        cv2.imshow(cam_id, frame)
                
                # waitKey es obligatorio para renderizar la interfaz OpenCV y debe ir en el main thread
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: # 'q' o Esc para salir
                    print("\n[SISTEMA] Solicitud de cierre recibida. Deteniendo hilos...")
                    stop_event.set()
                    break
            else:
                # Si es modo headless, solo dormimos un segundo para no consumir CPU
                time.sleep(1.0)
                
    except KeyboardInterrupt:
        print("\n[!] Interrupcion de teclado (Ctrl+C) recibida. Deteniendo hilos...")
        stop_event.set()

    # 5. Esperar a que todos los hilos terminen y limpiar recursos
    print("\nEsperando la detencion de trabajadores de camaras...")
    for worker in workers:
        worker.join(timeout=3.0)

    cv2.destroyAllWindows()
    db.cerrar()
    print("=================================================================")
    print("              SISTEMA DE SEGURIDAD APAGADO CON EXITO             ")
    print("=================================================================")

if __name__ == "__main__":
    main()
