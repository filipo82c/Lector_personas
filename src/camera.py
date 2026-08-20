import os
import time
import datetime
import threading
import cv2
import config

class FrameGrabber(threading.Thread):
    """Subproceso dedicado a leer frames del hardware a maxima velocidad (30 FPS) sin bloqueos."""
    def __init__(self, cap, worker):
        super().__init__()
        self.cap = cap
        self.worker = worker
        self.daemon = True
        
    def run(self):
        conteo_fallos = 0
        while self.worker.running and not self.worker.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                conteo_fallos = 0
                
                # Voltear horizontal si esta configurado (util para webcams)
                if config.SISTEMA.get("voltear_horizontal", True):
                    frame = cv2.flip(frame, 1)
                    
                self.worker.raw_frame = frame
            else:
                # Si es un archivo de video local y llego al final, rebobinarlo para reproduccion continua
                if isinstance(self.worker.source, str) and os.path.exists(self.worker.source):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.03)
                    continue

                conteo_fallos += 1
                if conteo_fallos == 10:
                    print(f"[!] [CAMARA {self.worker.camera_id}] No se pudo obtener imagen desde la fuente '{self.worker.source}'.")
                    print("    Tip: Si no tienes camara física conectada, puedes colocar la ruta a un archivo .mp4 en config.json.")
                    self.worker.reconnect_needed = True
                time.sleep(0.5)

class InferenceThread(threading.Thread):
    """Subproceso dedicado exclusivamente a ejecutar la inferencia de IA en segundo plano."""
    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.daemon = True
        
    def run(self):
        # Tasa de inferencia controlada para no sobrecargar el procesador (unas 12 veces por segundo)
        # 12 FPS de IA es ideal para vigilancia en tiempo real, mientras el video se muestra a 30 FPS.
        fps_ia = 12.0
        intervalo = 1.0 / fps_ia
        
        while self.worker.running and not self.worker.stop_event.is_set():
            t_inicio = time.time()
            frame = self.worker.raw_frame
            
            if frame is not None:
                # Ejecutar inferencia completa (skip_inference=False)
                # Esta llamada actualizara el cache interno detector.last_results[camera_id]
                annotated_frame, eventos = self.worker.detector.process_frame(
                    frame, camara_id=self.worker.camera_id, skip_inference=False
                )
                
                # Procesar eventos en base de datos (SQLite / Firebase)
                if eventos:
                    self.worker.procesar_eventos(frame, annotated_frame, eventos)
                    
            # Controlar tasa de ejecucion
            t_fin = time.time()
            procesamiento = t_fin - t_inicio
            espera = max(0.001, intervalo - procesamiento)
            time.sleep(espera)

class CameraWorker(threading.Thread):
    """Trabajador principal que orquesta la lectura de frames y los renderiza en un loop estable de 30 FPS."""
    
    def __init__(self, camera_id, source, detector, db, latest_frames, stop_event):
        super().__init__()
        self.camera_id = camera_id
        self.source = source
        self.detector = detector
        self.db = db
        self.latest_frames = latest_frames
        self.stop_event = stop_event
        
        self.daemon = True
        self.running = True
        self.reconnect_needed = False
        
        # Almacenamiento del ultimo frame leido por el grabador
        self.raw_frame = None
        
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
        # Intentar solicitar resolución HD / Full HD al dispositivo de cámara
        if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
        if isinstance(self.source, str) and self.source.startswith("rtsp"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        # Iniciar hilos auxiliares
        self.grabber = FrameGrabber(cap, self)
        self.grabber.start()
        
        self.inference = InferenceThread(self)
        self.inference.start()
        
        # Loop principal de renderizado y visualizacion estable a 30 FPS
        # Este hilo no hace inferencia pesada, por lo que nunca bajara de 30 FPS.
        fps_objetivo = 30.0
        intervalo_frame = 1.0 / fps_objetivo
        
        while not self.stop_event.is_set():
            t_inicio = time.time()
            
            # Reconectar si el grabador indico que el flujo se cayo
            if self.reconnect_needed:
                print(f"[CAMARA {self.camera_id}] Flujo caido. Reabriendo video...")
                self.reconnect_needed = False
                cap.release()
                cap = cv2.VideoCapture(self.source)
                self.grabber = FrameGrabber(cap, self)
                self.grabber.start()
                time.sleep(1.0)
                continue
                
            frame = self.raw_frame
            if frame is not None:
                # Copiar y pintar los ultimos cuadros de deteccion sobre el frame actual (cero latencia)
                annotated_frame = frame.copy()
                cache = self.detector.last_results.get(self.camera_id)
                if cache is not None:
                    self.detector.dibujar_resultados_cacheados(annotated_frame, cache)
                
                # Publicar el frame para que main.py lo muestre
                self.latest_frames[self.camera_id] = annotated_frame
                
            # Control de tiempo para garantizar exactamente 30 FPS
            t_fin = time.time()
            procesamiento = t_fin - t_inicio
            espera = max(0.001, intervalo_frame - procesamiento)
            time.sleep(espera)
            
        # Detener hilos
        self.running = False
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
