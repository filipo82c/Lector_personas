import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
import config

class SecurityDetector:
    """Clase unificada que maneja YOLO11 (personas y armas) y OpenCV YuNet/SFace (rostros)."""
    
    def __init__(self):
        self.use_opencl = config.DETECTOR.get("use_opencl", True)
        
        # 1. Cargar Modelos de Rostros (OpenCV DNN)
        self.inicializar_modelos_rostros()
        
        # 2. Cargar Modelos YOLO (Ultralytics)
        print("Cargando modelo YOLO11 base (Personas)...")
        self.yolo_base = YOLO(config.DETECTOR["yolo_base_path"])
        
        print("Cargando modelo YOLO de amenazas (Armas)...")
        self.yolo_threat = YOLO(config.DETECTOR["yolo_threat_path"])
        
        # Clases del modelo de amenazas (Subh775/Threat-Detection-YOLOv8n)
        # 0: Gun, 1: Explosive, 2: Grenade, 3: Knife
        self.threat_labels = {
            0: "Arma de Fuego",
            1: "Explosivo",
            2: "Granada",
            3: "Arma Blanca"
        }
        
        # 3. Base de Datos de Rostros
        self.db_embeddings = {}
        self.cargar_base_datos_rostros()
        
    def inicializar_modelos_rostros(self):
        """Inicializa los detectores y reconocedores faciales con aceleracion OpenCL si esta activa."""
        yunet_path = config.DETECTOR["yunet_path"]
        sface_path = config.DETECTOR["sface_path"]
        det_threshold = config.DETECTOR["face_det_threshold"]
        
        backend = cv2.dnn.DNN_BACKEND_OPENCV
        
        if self.use_opencl:
            target = cv2.dnn.DNN_TARGET_OPENCL
            print("Intentando inicializar modelos faciales con aceleracion GPU (OpenCL)...")
        else:
            target = cv2.dnn.DNN_TARGET_CPU
            print("Inicializando modelos faciales en CPU...")
            
        try:
            # Usar argumentos posicionales para maxima compatibilidad en OpenCV 4.x y 5.x
            self.detector_face = cv2.FaceDetectorYN.create(
                yunet_path, "", (320, 320), det_threshold, 0.3, 5000, backend, target
            )
            
            self.recognizer_face = cv2.FaceRecognizerSF.create(
                sface_path, "", backend, target
            )
            print("[OK] Modelos de rostros cargados exitosamente.")
        except Exception as e:
            if self.use_opencl:
                print(f"[!] Fallo la inicializacion con OpenCL: {e}. Reintentando con CPU...")
                self.use_opencl = False
                self.inicializar_modelos_rostros()
            else:
                raise RuntimeError(f"Error critico al cargar modelos faciales: {e}")
                
    def cargar_base_datos_rostros(self):
        """Carga los rostros de db_rostros/, calcula embeddings y los cachea en un archivo JSON."""
        embeddings_json_path = os.path.join(config.DB_DIR, "embeddings.json")
        
        # Escanear archivos de imagen en la carpeta
        formatos_validos = ('.jpg', '.jpeg', '.png', '.webp')
        imagenes_locales = {}
        
        for item in os.listdir(config.DB_DIR):
            item_path = os.path.join(config.DB_DIR, item)
            # Soporte para archivos directos: Juan_Perez.jpg
            if os.path.isfile(item_path) and item.lower().endswith(formatos_validos):
                nombre = os.path.splitext(item)[0].replace("_", " ")
                imagenes_locales[nombre] = item_path
            # Soporte para subcarpetas: Juan_Perez/foto1.jpg
            elif os.path.isdir(item_path):
                nombre = item.replace("_", " ")
                # Buscar primera imagen valida en la subcarpeta
                for sub_item in os.listdir(item_path):
                    if sub_item.lower().endswith(formatos_validos):
                        imagenes_locales[nombre] = os.path.join(item_path, sub_item)
                        break
                        
        if not imagenes_locales:
            print("[DB ROSTROS] No se encontraron imagenes de personas registradas. La base de datos esta vacia.")
            return

        # Intentar cargar cache JSON
        cache_valido = False
        if os.path.exists(embeddings_json_path):
            try:
                with open(embeddings_json_path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                
                # Verificar si los nombres coinciden exactamente
                if set(cache_data.keys()) == set(imagenes_locales.keys()):
                    self.db_embeddings = {name: np.array(emb, dtype=np.float32) for name, emb in cache_data.items()}
                    print(f"[DB ROSTROS] Cargadas {len(self.db_embeddings)} firmas faciales desde el cache JSON.")
                    cache_valido = True
            except Exception as e:
                print(f"[DB ROSTROS] Error al leer embeddings.json: {e}. Se recalcularan los embeddings.")
                
        if not cache_valido:
            print("[DB ROSTROS] Calculando firmas faciales (embeddings)...")
            self.db_embeddings = {}
            for nombre, img_path in imagenes_locales.items():
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                # Extraer embedding
                h, w = img.shape[:2]
                self.detector_face.setInputSize((w, h))
                ret, faces = self.detector_face.detect(img)
                
                if ret and faces is not None:
                    # Usar el rostro con mayor confianza (el primero)
                    face_box = faces[0]
                    aligned = self.recognizer_face.alignCrop(img, face_box)
                    feat = self.recognizer_face.feature(aligned)
                    self.db_embeddings[nombre] = feat.flatten()
                    print(f"[DB ROSTROS] Rostro procesado con exito: {nombre}")
                else:
                    print(f"[DB ROSTROS] [WARNING] No se detecto rostro en la foto de: {nombre}")
            
            # Guardar en cache JSON
            try:
                cache_to_save = {name: emb.tolist() for name, emb in self.db_embeddings.items()}
                with open(embeddings_json_path, "w", encoding="utf-8") as f:
                    json.dump(cache_to_save, f, indent=4, ensure_ascii=False)
                print("[DB ROSTROS] Cache JSON de embeddings guardado exitosamente.")
            except Exception as e:
                print(f"[DB ROSTROS] Error al guardar embeddings.json: {e}")
                
    def process_frame(self, frame, camara_id="Camara"):
        """Procesa un frame para detectar personas (cuerpo), rostros, reconocer identidades y buscar armas."""
        # Clon para pintar encima
        draw_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Inicializar eventos de este frame
        eventos = []
        
        # 1. DETECCION DE ROSTROS Y RECONOCIMIENTO (YuNet + SFace)
        self.detector_face.setInputSize((w, h))
        ret_face, faces = self.detector_face.detect(frame)
        
        if ret_face and faces is not None:
            cosine_similarity_type = cv2.FaceRecognizerSF_FR_COSINE if hasattr(cv2, 'FaceRecognizerSF_FR_COSINE') else 0
            rec_threshold = config.DETECTOR["face_rec_threshold"]
            
            for face in faces:
                # Coordenadas de la caja del rostro
                fx, fy, fw, fh = map(int, face[0:4])
                conf_det = face[14]
                
                # Alinear y extraer embedding
                aligned = self.recognizer_face.alignCrop(frame, face)
                live_feat = self.recognizer_face.feature(aligned).flatten()
                
                # Comparar con la Base de Datos
                mejor_nombre = "Desconocido"
                mejor_score = -1.0
                
                for nombre, db_feat in self.db_embeddings.items():
                    score = self.recognizer_face.match(live_feat, db_feat, cosine_similarity_type)
                    if score > mejor_score:
                        mejor_score = score
                        mejor_nombre = nombre
                        
                # Verificar si supera el umbral de similitud
                es_reconocido = mejor_score >= rec_threshold
                label_name = mejor_nombre if es_reconocido else "Desconocido"
                conf_final = mejor_score if es_reconocido else conf_det
                
                # Dibujar caja del rostro
                # Verde para reconocido, Amarillo para desconocido
                color_rostro = (0, 255, 0) if es_reconocido else (0, 255, 255)
                cv2.rectangle(draw_frame, (fx, fy), (fx + fw, fy + fh), color_rostro, 2)
                
                # Etiqueta de texto
                txt = f"{label_name} ({mejor_score:.2f})" if es_reconocido else "Desconocido"
                cv2.putText(draw_frame, txt, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rostro, 2)
                
                # Registrar evento
                eventos.append({
                    "tipo": "acceso",
                    "persona": label_name,
                    "confianza": float(conf_final),
                    "es_reconocido": es_reconocido,
                    "bbox_rostro": [fx, fy, fw, fh]
                })

        # 2. DETECCION DE CUERPO (YOLO11)
        res_base = self.yolo_base(frame, verbose=False)[0]
        yolo_person_threshold = config.DETECTOR["yolo_person_threshold"]
        
        for box in res_base.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Clase 0 = Person en COCO
            if cls_id == 0 and conf >= yolo_person_threshold:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                # Dibujar caja azul de persona
                cv2.rectangle(draw_frame, (bx1, by1), (bx2, by2), (255, 120, 0), 2)
                cv2.putText(draw_frame, "Persona", (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 1)
                
        # 3. DETECCION DE AMENAZAS (YOLO11 / YOLOv8 Threat)
        res_threat = self.yolo_threat(frame, verbose=False)[0]
        yolo_threat_threshold = config.DETECTOR["yolo_threat_threshold"]
        
        for box in res_threat.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Verificar si la clase es una amenaza
            if cls_id in self.threat_labels and conf >= yolo_threat_threshold:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                amenaza_nombre = self.threat_labels[cls_id]
                
                # Alerta detectada! Dibujar caja roja intermitente o gruesa
                cv2.rectangle(draw_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
                cv2.putText(draw_frame, f"ALERTA: {amenaza_nombre} ({conf:.2f})", (bx1, by1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                eventos.append({
                    "tipo": "alerta",
                    "detalle": amenaza_nombre,
                    "confianza": float(conf),
                    "bbox_amenaza": [bx1, by1, bx2 - bx1, by2 - by1]
                })
                
        return draw_frame, eventos
