import os
import sys
import time
import threading
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import database
import detector
import camera

class SecurityEngine:
    """Motor de IA y vigilancia desacoplado que puede ser controlado por la GUI o acoplado a otros programas."""
    
    def __init__(self):
        self.running = False
        self.stop_event = threading.Event()
        self.latest_frames = {} # {camera_id: frame}
        self.workers = []
        self.db = None
        self.detector_ia = None
        
        # Callbacks para integración con la GUI o programas externos
        self.alert_callbacks = []
        self.access_callbacks = []
        
        self.inicializar_motores()
        
    def inicializar_motores(self):
        """Carga la base de datos y los modelos de IA una sola vez."""
        print("[ENGINE] Inicializando base de datos...")
        self.db = database.obtener_base_datos()
        
        print("[ENGINE] Cargando motores de IA (YOLO + YuNet + SFace)...")
        self.detector_ia = detector.SecurityDetector()
        print(f"[ENGINE OK] Motores listos. Identidades registradas: {len(self.detector_ia.db_embeddings)}")

    def register_alert_callback(self, callback_fn):
        """Registra una función callback que será llamada al detectar una amenaza."""
        if callback_fn not in self.alert_callbacks:
            self.alert_callbacks.append(callback_fn)

    def register_access_callback(self, callback_fn):
        """Registra una función callback que será llamada al reconocer a una persona."""
        if callback_fn not in self.access_callbacks:
            self.access_callbacks.append(callback_fn)

    def start_surveillance(self):
        """Inicia el procesamiento de cámaras en hilos secundarios."""
        if self.running:
            print("[ENGINE] La vigilancia ya está en ejecución.")
            return True
            
        print("[ENGINE] Arrancando hilos de cámaras...")
        self.stop_event.clear()
        self.running = True
        self.workers = []
        
        for cam_info in config.CAMARAS:
            cam_id = cam_info["id"]
            source = cam_info["source"]
            
            worker = CustomCameraWorker(
                camera_id=cam_id,
                source=source,
                detector=self.detector_ia,
                db=self.db,
                latest_frames=self.latest_frames,
                stop_event=self.stop_event,
                engine=self
            )
            worker.start()
            self.workers.append(worker)
            
        print("[ENGINE OK] Vigilancia iniciada exitosamente.")
        return True

    def stop_surveillance(self):
        """Detiene el procesamiento de cámaras y libera recursos."""
        if not self.running:
            return True
            
        print("[ENGINE] Deteniendo vigilancia...")
        self.running = False
        self.stop_event.set()
        
        for w in self.workers:
            w.join(timeout=2.0)
            
        self.workers = []
        self.latest_frames.clear()
        print("[ENGINE OK] Vigilancia detenida limpiamente.")
        return True

    def is_running(self):
        return self.running

    def set_threat_threshold(self, threshold):
        """Ajusta el umbral de sensibilidad de detección de amenazas en caliente."""
        config.DETECTOR["yolo_threat_threshold"] = threshold
        print(f"[ENGINE] Umbral de amenazas ajustado a: {threshold}")

    def cerrar(self):
        """Cierre final del motor y base de datos."""
        self.stop_surveillance()
        if self.db:
            self.db.cerrar()
        print("[ENGINE] Recursos liberados totalmente.")

class CustomCameraWorker(camera.CameraWorker):
    """Subclase de CameraWorker que notifica los callbacks del engine."""
    def __init__(self, camera_id, source, detector, db, latest_frames, stop_event, engine):
        super().__init__(camera_id, source, detector, db, latest_frames, stop_event)
        self.engine = engine

    def procesar_eventos(self, raw_frame, annotated_frame, eventos):
        super().procesar_eventos(raw_frame, annotated_frame, eventos)
        
        # Disparar callbacks externos
        for ev in eventos:
            tipo = ev.get("tipo")
            if tipo == "acceso":
                for cb in self.engine.access_callbacks:
                    try:
                        cb(self.camera_id, ev, annotated_frame)
                    except Exception as e:
                        print(f"[ENGINE CALLBACK ERROR] {e}")
            elif tipo == "alerta":
                for cb in self.engine.alert_callbacks:
                    try:
                        cb(self.camera_id, ev, annotated_frame)
                    except Exception as e:
                        print(f"[ENGINE CALLBACK ERROR] {e}")
