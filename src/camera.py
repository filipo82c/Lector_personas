import os
import time
import datetime
import threading
import cv2
import config

class CameraWorker(threading.Thread):
    """Trabajador en segundo plano para capturar, procesar y guardar eventos de una camara."""
    
    def __init__(self, camera_id, source, detector, db, latest_frames, stop_event):
        super().__init__()
        self.camera_id = camera_id
        self.source = source
        self.detector = detector
        self.db = db
        self.latest_frames = latest_frames
        self.stop_event = stop_event
        
        self.daemon = True # El hilo finalizara si el programa principal se cierra
        
        # Tiempos de control para evitar spam en la base de datos (cool-downs)
        self.last_seen_times = {}    # {persona: timestamp}
        self.last_threat_times = {}  # {tipo_amenaza: timestamp}
        self.last_unknown_time = 0   # timestamp
        
        # Tiempos de espera configurables en segundos
        self.cooldown_persona = 15.0
        self.cooldown_amenaza = 8.0
        self.cooldown_desconocido = 10.0
        
    def run(self):
        print(f"[CAMARA {self.camera_id}] Iniciando captura de flujo desde: {self.source}")
        
        cap = cv2.VideoCapture(self.source)
        
        # Ajustar buffer para reducir latencia (especialmente util en camaras RTSP/IP)
        if isinstance(self.source, str) and self.source.startswith("rtsp"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        conteo_fallos = 0
        frame_count = 0
        
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            
            if not ret:
                conteo_fallos += 1
                print(f"[CAMARA {self.camera_id}] Error al leer frame (Fallo #{conteo_fallos}). Reintentando...")
                time.sleep(2.0)
                
                # Intentar reconectar si hay muchos fallos consecutivos
                if conteo_fallos >= 5:
                    print(f"[CAMARA {self.camera_id}] Reabriendo flujo de video...")
                    cap.release()
                    cap = cv2.VideoCapture(self.source)
                    conteo_fallos = 0
                continue
                
            conteo_fallos = 0
            frame_count += 1
            
            # Realizar inferencia completa cada 3 frames (multiplica por 3 la velocidad del video)
            skip_inference = (frame_count % 3 != 0)
            
            # Inferencia IA en el frame
            annotated_frame, eventos = self.detector.process_frame(
                frame, camara_id=self.camera_id, skip_inference=skip_inference
            )
            
            # Guardar el frame procesado en la memoria compartida para visualizacion
            self.latest_frames[self.camera_id] = annotated_frame
            
            # Procesar eventos y logs
            self.procesar_eventos(frame, annotated_frame, eventos)
            
            # Controlar tasa de captura para no saturar la CPU
            time.sleep(0.01)
            
        cap.release()
        print(f"[CAMARA {self.camera_id}] Flujo de video cerrado limpiamente.")
        
    def procesar_eventos(self, raw_frame, annotated_frame, eventos):
        """Revisa los resultados obtenidos, guarda fotos de alerta y registra en BD con cool-downs."""
        ahora = time.time()
        ahora_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for ev in eventos:
            tipo = ev["tipo"]
            
            # 1. Procesamiento de Reconocimiento Facial (acceso)
            if tipo == "acceso":
                persona = ev["persona"]
                confianza = ev["confianza"]
                es_reconocido = ev["es_reconocido"]
                
                if es_reconocido:
                    # Aplicar cool-down para no registrar a la misma persona en cada frame
                    ultimo_registro = self.last_seen_times.get(persona, 0)
                    if ahora - ultimo_registro >= self.cooldown_persona:
                        self.db.registrar_acceso(self.camera_id, persona, confianza)
                        self.last_seen_times[persona] = ahora
                else:
                    # Persona Desconocida en area de seguridad privada
                    if ahora - self.last_unknown_time >= self.cooldown_desconocido:
                        self.db.registrar_acceso(self.camera_id, "Desconocido", confianza)
                        self.last_unknown_time = ahora
                        
                        # Guardar captura si esta configurado
                        if config.SISTEMA.get("guardar_alertas_desconocidos", False):
                            filename = f"INTRUSO_{self.camera_id}_{ahora_str}.jpg"
                            filepath = os.path.join(config.ALERT_DIR, filename)
                            cv2.imwrite(filepath, annotated_frame)
                            print(f"[ALERTA] Captura de intruso guardada en: {filepath}")
                            
            # 2. Procesamiento de Deteccion de Amenazas (armas)
            elif tipo == "alerta":
                detalle = ev["detalle"]
                confianza = ev["confianza"]
                
                # Cooldown de alerta por tipo de arma
                ultimo_registro = self.last_threat_times.get(detalle, 0)
                if ahora - ultimo_registro >= self.cooldown_amenaza:
                    # Ruta donde se guardara la evidencia
                    filepath = None
                    if config.SISTEMA.get("guardar_capturas_alertas", True):
                        filename = f"AMENAZA_{self.camera_id}_{detalle.replace(' ', '_')}_{ahora_str}.jpg"
                        filepath = os.path.join(config.ALERT_DIR, filename)
                        # Guardar el frame anotado para ver la evidencia marcada por la IA
                        cv2.imwrite(filepath, annotated_frame)
                        print(f"[ALERTA CRITICA] Evidencia guardada en: {filepath}")
                        
                    # Registrar alerta en Base de Datos (SQLite + Nube)
                    self.db.registrar_alerta(self.camera_id, detalle, "alta", filepath)
                    self.last_threat_times[detalle] = ahora
