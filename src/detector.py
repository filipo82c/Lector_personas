import os
import json
import zipfile
import cv2
import numpy as np
import time
from collections import Counter
from ultralytics import YOLO
import config

class SecurityDetector:
    """Clase unificada que maneja YOLO11 (personas y armas) y OpenCV YuNet/SFace (rostros)."""
    
    def __init__(self):
        self.use_opencl = config.DETECTOR.get("use_opencl", True)
        self.target_inference_width = 640 # Redimensionar para inferencia ultra rapida
        self.last_cache_save_time = 0
        
        # Rastreabilidad de rostros para estabilización temporal (elimina parpadeos)
        self.face_tracks = {} # {track_id: { "centroid": (cx, cy), "history": [nombres], "frames_since_update": 0 }}
        self.next_track_id = 0
        
        # Cache de detecciones previas por camara (para saltar frames sin parpadeos)
        self.last_results = {} # {camara_id: { "accesos": [...], "persons": [...], "threats": [...] }}
        
        # 1. Cargar Modelos de Rostros (OpenCV DNN)
        self.inicializar_modelos_rostros()
        
        # 2. Cargar Modelos YOLO (Ultralytics)
        print("Cargando modelo YOLO11 base (Personas)...")
        self.yolo_base = YOLO(config.DETECTOR["yolo_base_path"])
        
        print("Cargando modelo YOLO de amenazas (Armas)...")
        threat_path = config.DETECTOR["yolo_threat_path"]
        
        # Soporte automatico por si el modelo proviene de un archivo ZIP de Google Drive / Colab
        if os.path.exists(threat_path) and (threat_path.endswith(".zip") or zipfile.is_zipfile(threat_path)):
            print(f"[!] Se detecto que '{os.path.basename(threat_path)}' es un archivo comprimido ZIP. Extrayendo pesos .pt...")
            temp_extract_dir = os.path.join(config.MODEL_DIR, "_tmp_zip")
            os.makedirs(temp_extract_dir, exist_ok=True)
            with zipfile.ZipFile(threat_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)
            
            # Buscar el archivo .pt extraido (ej: best.pt o weights/best.pt)
            pt_encontrado = None
            for root, _, files in os.walk(temp_extract_dir):
                for f in files:
                    if f.endswith(".pt"):
                        pt_encontrado = os.path.join(root, f)
                        break
                if pt_encontrado:
                    break
            
            if pt_encontrado:
                target_pt = os.path.join(config.MODEL_DIR, "threat_detection.pt")
                if os.path.exists(target_pt) and os.path.abspath(target_pt) != os.path.abspath(pt_encontrado):
                    os.remove(target_pt)
                os.replace(pt_encontrado, target_pt)
                threat_path = target_pt
                config.DETECTOR["yolo_threat_path"] = target_pt
                print(f"[OK] Pesos .pt extraidos exitosamente a: {target_pt}")

        self.yolo_threat = YOLO(threat_path)
        
        # Obtener clases dinámicamente del modelo YOLO y traducirlas si aplica
        self.threat_labels = {}
        for idx, name in self.yolo_threat.names.items():
            name_lower = name.lower()
            if name_lower in ["gun", "guns", "pistol", "rifle", "firearm"]:
                self.threat_labels[idx] = "Arma de Fuego"
            elif name_lower in ["knife", "knives", "blade", "sword", "weapon"]:
                self.threat_labels[idx] = "Arma Blanca"
            elif name_lower in ["explosive", "explosives", "bomb"]:
                self.threat_labels[idx] = "Explosivo"
            elif name_lower in ["grenade", "grenades"]:
                self.threat_labels[idx] = "Granada"
            else:
                self.threat_labels[idx] = name.capitalize()
        
        # 3. Base de Datos de Rostros
        self.db_embeddings = {} # nombre -> [lista de embeddings]
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
            # YuNet requiere un tamaño de entrada inicial
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
        imagenes_locales = {} # nombre -> [lista de rutas]
        
        for item in os.listdir(config.DB_DIR):
            item_path = os.path.join(config.DB_DIR, item)
            # Soporte para archivos directos: Filip_Sanabria.png, Filip_Sanabria_lentes.png
            if os.path.isfile(item_path) and item.lower().endswith(formatos_validos):
                # Extraer nombre base quitando sufijos despues del segundo guion bajo si es un patron
                partes = os.path.splitext(item)[0].split("_")
                if len(partes) >= 2:
                    nombre = f"{partes[0]} {partes[1]}"
                else:
                    nombre = partes[0]
                    
                if nombre not in imagenes_locales:
                    imagenes_locales[nombre] = []
                imagenes_locales[nombre].append(item_path)
                
            # Soporte para subcarpetas: Filip_Sanabria/foto1.png
            elif os.path.isdir(item_path):
                nombre = item.replace("_", " ")
                if nombre not in imagenes_locales:
                    imagenes_locales[nombre] = []
                for sub_item in os.listdir(item_path):
                    if sub_item.lower().endswith(formatos_validos):
                        imagenes_locales[nombre].append(os.path.join(item_path, sub_item))
                        
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
                    self.db_embeddings = {}
                    for name, embs in cache_data.items():
                        # Si es un cache antiguo (lista simple de floats en lugar de lista de listas), envolverlo
                        if len(embs) > 0 and not isinstance(embs[0], list):
                            embs = [embs]
                        self.db_embeddings[name] = [np.array(e, dtype=np.float32) for e in embs]
                    print(f"[DB ROSTROS] Cargadas {len(self.db_embeddings)} identidades con multiples firmas desde el cache JSON.")
                    cache_valido = True
            except Exception as e:
                print(f"[DB ROSTROS] Error al leer embeddings.json: {e}. Se recalcularan los embeddings.")
                
        if not cache_valido:
            print("[DB ROSTROS] Calculando firmas faciales (embeddings)...")
            self.db_embeddings = {}
            for nombre, lista_rutas in imagenes_locales.items():
                self.db_embeddings[nombre] = []
                for img_path in lista_rutas:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                        
                    h, w = img.shape[:2]
                    self.detector_face.setInputSize((w, h))
                    ret, faces = self.detector_face.detect(img)
                    
                    if ret and faces is not None:
                        # Usar el rostro con mayor confianza (el primero)
                        face_box = faces[0]
                        aligned = self.recognizer_face.alignCrop(img, face_box)
                        feat = self.recognizer_face.feature(aligned)
                        self.db_embeddings[nombre].append(feat.flatten())
                        print(f"[DB ROSTROS] Firma facial extraida para {nombre} desde: {os.path.basename(img_path)}")
                    else:
                        print(f"[DB ROSTROS] [WARNING] No se detecto rostro en la foto de: {nombre} ({os.path.basename(img_path)})")
                
                # Limpiar si no se logro extraer ningun embedding para esta persona
                if not self.db_embeddings[nombre]:
                    del self.db_embeddings[nombre]
            
            # Guardar en cache JSON
            self.guardar_cache_embeddings()

    def guardar_cache_embeddings(self):
        """Guarda la base de datos de embeddings actual en el cache JSON."""
        embeddings_json_path = os.path.join(config.DB_DIR, "embeddings.json")
        try:
            cache_to_save = {name: [emb.tolist() for emb in embs] for name, embs in self.db_embeddings.items()}
            with open(embeddings_json_path, "w", encoding="utf-8") as f:
                json.dump(cache_to_save, f, indent=4, ensure_ascii=False)
            print("[DB ROSTROS] Cache JSON de embeddings guardado exitosamente.")
        except Exception as e:
            print(f"[DB ROSTROS] Error al guardar embeddings.json: {e}")
                
    def process_frame(self, frame, camara_id="Camara", skip_inference=False):
        """Procesa un frame. Si skip_inference=True, dibuja los resultados cacheados para mantener altos FPS."""
        draw_frame = frame.copy()
        
        # Si se solicita omitir inferencia y tenemos resultados previos, dibujarlos de inmediato
        if skip_inference and camara_id in self.last_results:
            self.dibujar_resultados_cacheados(draw_frame, self.last_results[camara_id])
            return draw_frame, [] # No se emiten nuevos eventos en frames omitidos
            
        h, w = frame.shape[:2]
        
        # Redimensionar para acelerar la inferencia (Optimización crítica de CPU)
        if w > self.target_inference_width:
            ratio = self.target_inference_width / w
            rw = self.target_inference_width
            rh = int(h * ratio)
            scale_x = w / rw
            scale_y = h / rh
            resized_frame = cv2.resize(frame, (rw, rh))
        else:
            rw, rh = w, h
            scale_x = 1.0
            scale_y = 1.0
            resized_frame = frame

        eventos = []
        
        # Estructura del cache de este frame
        cache_frame = {
            "persons": [],  # [ [x1, y1, x2, y2], ... ]
            "accesos": [],  # [ {"box": [fx, fy, fw, fh], "label": label, "color": color}, ... ]
            "threats": []   # [ {"box": [bx1, by1, bx2, by2], "label": label, "conf": conf}, ... ]
        }

        # 1. DETECCION DE ROSTROS Y RECONOCIMIENTO (YuNet + SFace)
        self.detector_face.setInputSize((rw, rh))
        ret_face, faces = self.detector_face.detect(resized_frame)
        
        nuevos_tracks = {}
        
        if ret_face and faces is not None:
            cosine_similarity_type = cv2.FaceRecognizerSF_FR_COSINE if hasattr(cv2, 'FaceRecognizerSF_FR_COSINE') else 0
            rec_threshold = config.DETECTOR["face_rec_threshold"]
            candidatos_frame = []
            
            for f_idx, face in enumerate(faces):
                fx, fy, fw, fh = map(int, face[:4])
                fx_orig = int(fx * scale_x)
                fy_orig = int(fy * scale_y)
                fw_orig = int(fw * scale_x)
                fh_orig = int(fh * scale_y)
                
                cx = fx_orig + fw_orig / 2
                cy = fy_orig + fh_orig / 2
                
                # Tracking
                min_dist = float('inf')
                match_id = None
                for tid, track in self.face_tracks.items():
                    tcx, tcy = track["centroid"]
                    dist = np.hypot(cx - tcx, cy - tcy)
                    if dist < min_dist:
                        min_dist = dist
                        match_id = tid
                        
                if match_id is not None and min_dist < 120:
                    tid = match_id
                    track_data = self.face_tracks[tid]
                    track_data["centroid"] = (cx, cy)
                    track_data["frames_since_update"] = 0
                else:
                    tid = self.next_track_id
                    self.next_track_id += 1
                    track_data = {
                        "centroid": (cx, cy),
                        "history": [],
                        "frames_since_update": 0
                    }
                
                face_scaled = face.copy()
                face_scaled[0] = fx_orig
                face_scaled[1] = fy_orig
                face_scaled[2] = fw_orig
                face_scaled[3] = fh_orig
                for idx in range(4, 14, 2):
                    face_scaled[idx] = face[idx] * scale_x
                    face_scaled[idx+1] = face[idx+1] * scale_y
                
                aligned = self.recognizer_face.alignCrop(frame, face_scaled)
                live_feat = self.recognizer_face.feature(aligned).flatten()
                
                # Calcular puntaje contra cada persona de la base de datos estricta
                scores_persona = {}
                for nombre, embs in self.db_embeddings.items():
                    max_s = -1.0
                    for db_feat in embs:
                        if not isinstance(db_feat, np.ndarray) or db_feat.ndim == 0 or db_feat.size != 128:
                            continue
                        s = self.recognizer_face.match(live_feat, db_feat, cosine_similarity_type)
                        if s > max_s:
                            max_s = s
                    if max_s > -1.0:
                        scores_persona[nombre] = max_s
                        
                personas_ord = sorted(scores_persona.items(), key=lambda x: x[1], reverse=True)
                top_nombre = "Desconocido"
                top_score = -1.0
                segundo_score = -1.0
                
                if personas_ord:
                    top_nombre, top_score = personas_ord[0]
                    if len(personas_ord) > 1:
                        segundo_score = personas_ord[1][1]
                        
                margen = (top_score - segundo_score) if len(personas_ord) > 1 else 1.0
                
                candidatos_frame.append({
                    "f_idx": f_idx,
                    "tid": tid,
                    "track_data": track_data,
                    "box_orig": [fx_orig, fy_orig, fw_orig, fh_orig],
                    "top_nombre": top_nombre,
                    "top_score": top_score,
                    "margen": margen
                })
                
            # ORDENAR Y ASIGNACIÓN ÚNICA UNO-A-UNO (Ningún nombre se puede repetir en el mismo frame)
            candidatos_frame.sort(key=lambda x: x["top_score"], reverse=True)
            nombres_asignados = set()
            
            for c in candidatos_frame:
                tid = c["tid"]
                track_data = c["track_data"]
                fx_orig, fy_orig, fw_orig, fh_orig = c["box_orig"]
                top_nombre = c["top_nombre"]
                top_score = c["top_score"]
                margen = c["margen"]
                
                # REGLAS DE RECONOCIMIENTO ESTRICTO:
                # 1. Similitud >= 0.48 (Umbral estricto para SFace)
                # 2. Margen de distinción respecto al 2do candidato >= 0.08
                # 3. La identidad NO debe haber sido asignada ya a otro rostro en este frame
                es_valido = (top_score >= rec_threshold) and (margen >= 0.08) and (top_nombre not in nombres_asignados)
                
                label_raw = top_nombre if es_valido else "Desconocido"
                if es_valido:
                    nombres_asignados.add(top_nombre)
                    
                track_data["history"].append(label_raw)
                if len(track_data["history"]) > 7:
                    track_data["history"].pop(0)
                    
                mode_name = Counter(track_data["history"]).most_common(1)[0][0]
                color_rostro = (0, 255, 0) if mode_name != "Desconocido" else (0, 255, 255)
                
                cache_frame["accesos"].append({
                    "box": [fx_orig, fy_orig, fw_orig, fh_orig],
                    "label": f"{mode_name} ({top_score:.2f})" if mode_name != "Desconocido" else "Desconocido",
                    "color": color_rostro
                })
                
                eventos.append({
                    "tipo": "acceso",
                    "persona": mode_name,
                    "confianza": float(top_score if mode_name != "Desconocido" else 0.0),
                    "es_reconocido": mode_name != "Desconocido",
                    "bbox_rostro": [fx_orig, fy_orig, fw_orig, fh_orig]
                })
                
                nuevos_tracks[tid] = track_data

        # Mantener vivos los tracks que no se vieron en este frame por hasta 10 frames
        for tid, track in list(self.face_tracks.items()):
            if tid not in nuevos_tracks:
                track["frames_since_update"] += 1
                if track["frames_since_update"] < 10:
                    nuevos_tracks[tid] = track
        self.face_tracks = nuevos_tracks

        # 2. DETECCION DE CUERPO (YOLO11 sobre frame redimensionado para velocidad)
        res_base = self.yolo_base(resized_frame, verbose=False)[0]
        yolo_person_threshold = config.DETECTOR["yolo_person_threshold"]
        
        for box in res_base.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls_id == 0 and conf >= yolo_person_threshold:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                bx1_orig = int(bx1 * scale_x)
                by1_orig = int(by1 * scale_y)
                bx2_orig = int(bx2 * scale_x)
                by2_orig = int(by2 * scale_y)
                
                cache_frame["persons"].append([bx1_orig, by1_orig, bx2_orig, by2_orig])

        # 3. DETECCION DE AMENAZAS (YOLO11 Threat)
        res_threat = self.yolo_threat(resized_frame, verbose=False)[0]
        yolo_threat_threshold = config.DETECTOR["yolo_threat_threshold"]
        
        for box in res_threat.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls_id in self.threat_labels and conf >= yolo_threat_threshold:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                bx1_orig = int(bx1 * scale_x)
                by1_orig = int(by1 * scale_y)
                bx2_orig = int(bx2 * scale_x)
                by2_orig = int(by2 * scale_y)
                
                # Filtro de área mínima para evitar falsas alarmas por ruido o detalles diminutos
                w_box = bx2_orig - bx1_orig
                h_box = by2_orig - by1_orig
                if (w_box * h_box) < 900:
                    continue
                
                amenaza_nombre = self.threat_labels[cls_id]
                
                cache_frame["threats"].append({
                    "box": [bx1_orig, by1_orig, bx2_orig, by2_orig],
                    "label": f"ALERTA: {amenaza_nombre} ({conf:.2f})",
                    "conf": conf
                })
                
                eventos.append({
                    "tipo": "alerta",
                    "detalle": amenaza_nombre,
                    "confianza": float(conf),
                    "bbox_amenaza": [bx1_orig, by1_orig, bx2_orig - bx1_orig, by2_orig - by1_orig]
                })
                
        # Guardar en memoria de cache
        self.last_results[camara_id] = cache_frame
        
        # Dibujar resultados actuales
        self.dibujar_resultados_cacheados(draw_frame, cache_frame)
        
        return draw_frame, eventos
        
    def dibujar_resultados_cacheados(self, draw_frame, cache):
        """Dibuja de forma rapida las detecciones precalculadas sobre un frame para simular altos FPS."""
        # 1. Dibujar Cuerpos
        for box in cache["persons"]:
            cv2.rectangle(draw_frame, (box[0], box[1]), (box[2], box[3]), (255, 120, 0), 2)
            cv2.putText(draw_frame, "Persona", (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 1)
            
        # 2. Dibujar Rostros
        for face in cache["accesos"]:
            box = face["box"]
            cv2.rectangle(draw_frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), face["color"], 2)
            cv2.putText(draw_frame, face["label"], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face["color"], 2)
            
        # 3. Dibujar Amenazas (Armas)
        for threat in cache["threats"]:
            box = threat["box"]
            cv2.rectangle(draw_frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
            cv2.putText(draw_frame, threat["label"], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
