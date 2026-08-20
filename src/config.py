import os
import json

# Directorios base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "modelos")
DB_DIR = os.path.join(BASE_DIR, "db_rostros")
ALERT_DIR = os.path.join(BASE_DIR, "alertas")

CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

# Configuraciones por defecto
DEFAULT_CONFIG = {
    # Lista de camaras a procesar (pueden ser indices de webcam o URLs RTSP)
    "camaras": [
        {"id": "Camara_Web_0", "source": 0}
        # Ejemplo de camara RTSP profesional:
        # {"id": "Camara_Pasillo", "source": "rtsp://admin:12345@192.168.1.100:554/stream1"}
    ],
    
    # Configuracion de Base de Datos
    "database": {
        # Opciones: "sqlite" o "firebase"
        "type": "sqlite",
        # Ruta para SQLite local
        "sqlite_db_name": "seguridad.db",
        # Configuracion para Firebase Realtime Database
        "firebase_db_url": "https://tu-proyecto-firebase-default-rtdb.firebaseio.com/",
        "firebase_key_path": "firebase_key.json"
    },
    
    # Parametros de Deteccion
    "detector": {
        # Ruta de los modelos
        "yunet_path": os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx"),
        "sface_path": os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx"),
        "yolo_base_path": os.path.join(MODEL_DIR, "yolo11n.pt"),
        "yolo_threat_path": os.path.join(MODEL_DIR, "threat_detection.pt"),
        
        # Umbrales
        "face_det_threshold": 0.70,   # Confianza min para detectar rostro (subido de 0.6 para evitar caras falsas en objetos)
        "face_rec_threshold": 0.45,   # Umbral similitud de coseno para SFace (subido de 0.363 para mayor selectividad)
        "yolo_person_threshold": 0.5, # Confianza min para detectar cuerpo de persona
        "yolo_threat_threshold": 0.55, # Confianza min para detectar armas (ajustado de 0.65 a 0.55 para mayor sensibilidad en cuchillos)
        
        # Aceleracion por hardware (OpenCL para GPU AMD)
        "use_opencl": True
    },
    
    # Opciones del sistema
    "sistema": {
        "mostrar_ventanas": True,     # Abre ventanas cv2.imshow por cada camara
        "guardar_capturas_alertas": True, # Guarda fotos en /alertas si detecta arma o desconocido
        "guardar_alertas_desconocidos": False # Guarda fotos tambien cuando no reconoce el rostro
    }
}

# Cargar configuraciones reales
if not os.path.exists(CONFIG_PATH):
    # Si no existe, crear con valores por defecto
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
        print(f"[OK] Creado archivo de configuracion por defecto en: {os.path.basename(CONFIG_PATH)}")
    except Exception as e:
        print(f"[!] Error al crear config.json: {e}")
    config = DEFAULT_CONFIG
else:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        # Combinar con los valores por defecto por si faltan llaves
        for k, v in DEFAULT_CONFIG.items():
            if k not in config:
                config[k] = v
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if sub_k not in config[k]:
                        config[k][sub_k] = sub_v
    except Exception as e:
        print(f"[!] Error al leer config.json (usando valores por defecto): {e}")
        config = DEFAULT_CONFIG

# Hacer variables accesibles desde config
CAMARAS = config["camaras"]
DATABASE = config["database"]
DETECTOR = config["detector"]
SISTEMA = config["sistema"]

# Garantizar que las rutas de los modelos sean validas y multiplataforma (Linux / Windows / macOS)
for key in ["yunet_path", "sface_path", "yolo_base_path", "yolo_threat_path"]:
    path_val = DETECTOR.get(key, "")
    if not os.path.exists(path_val):
        filename = os.path.basename(path_val)
        candidate = os.path.join(MODEL_DIR, filename)
        if os.path.exists(candidate):
            DETECTOR[key] = candidate
        else:
            # Mantener la ruta base esperada si aun no esta descargado
            DETECTOR[key] = candidate

