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

import gui_app

def main():
    if "--cli" in sys.argv:
        print("=================================================================")
        print("    LECTOR DE PERSONAS - MODO CONSOLA / SEGUNDO PLANO (CLI)     ")
        print("=================================================================")
        # Lanzamiento legacy en consola
        import database, detector, camera
        db = database.obtener_base_datos()
        detector_ia = detector.SecurityDetector()
        latest_frames = {}
        stop_event = threading.Event()
        workers = []
        for cam_info in config.CAMARAS:
            w = camera.CameraWorker(cam_info["id"], cam_info["source"], detector_ia, db, latest_frames, stop_event)
            w.start()
            workers.append(w)
        try:
            while not stop_event.is_set():
                time.sleep(1.0)
        except KeyboardInterrupt:
            stop_event.set()
        for w in workers:
            w.join(timeout=2.0)
        db.cerrar()
    else:
        # Modo Por Defecto: Aplicación de Escritorio Nativa (GUI Desktop)
        gui_app.main()

if __name__ == "__main__":
    main()
