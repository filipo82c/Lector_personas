import os
import sqlite3
import datetime
from abc import ABC, abstractmethod
import config

class DatabaseAdapter(ABC):
    """Interfaz abstracta para conectores de Base de Datos en Tiempo Real."""
    
    @abstractmethod
    def registrar_acceso(self, camara_id: str, persona: str, confianza: float):
        """Registra el acceso o reconocimiento de una persona."""
        pass
        
    @abstractmethod
    def registrar_alerta(self, camara_id: str, tipo_alerta: str, severidad: str, ruta_imagen: str = None):
        """Registra una alerta de seguridad crítica (armas o intrusos)."""
        pass
        
    @abstractmethod
    def cerrar(self):
        """Cierra conexiones activas."""
        pass

class SQLiteDatabase(DatabaseAdapter):
    """Conector local SQLite (Por defecto)."""
    
    def __init__(self):
        self.db_path = os.path.join(config.BASE_DIR, config.DATABASE["sqlite_db_name"])
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.crear_tablas()
        
    def crear_tablas(self):
        with self.conn:
            # Tabla de accesos
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS accesos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    camara_id TEXT NOT NULL,
                    persona TEXT NOT NULL,
                    confianza REAL NOT NULL
                )
            """)
            # Tabla de alertas
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS alertas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    camara_id TEXT NOT NULL,
                    tipo_alerta TEXT NOT NULL,
                    severidad TEXT NOT NULL,
                    ruta_imagen TEXT
                )
            """)
            
    def registrar_acceso(self, camara_id: str, persona: str, confianza: float):
        ahora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO accesos (timestamp, camara_id, persona, confianza) VALUES (?, ?, ?, ?)",
                    (ahora, camara_id, persona, confianza)
                )
            print(f"[DB LOCAL] Acceso registrado: {persona} (Conf: {confianza:.2f}) en {camara_id}")
        except Exception as e:
            print(f"[DB ERROR] Error al registrar acceso en SQLite: {e}")
            
    def registrar_alerta(self, camara_id: str, tipo_alerta: str, severidad: str, ruta_imagen: str = None):
        ahora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO alertas (timestamp, camara_id, tipo_alerta, severidad, ruta_imagen) VALUES (?, ?, ?, ?, ?)",
                    (ahora, camara_id, tipo_alerta, severidad, ruta_imagen)
                )
            print(f"[DB LOCAL] [ALERTA] Alerta {severidad.upper()} registrada: {tipo_alerta} en {camara_id}")
        except Exception as e:
            print(f"[DB ERROR] Error al registrar alerta en SQLite: {e}")
            
    def cerrar(self):
        if self.conn:
            self.conn.close()

class FirebaseDatabase(DatabaseAdapter):
    """Conector en la nube con Firebase Realtime Database para sincronizacion en tiempo real."""
    
    def __init__(self):
        self.local_db = SQLiteDatabase() # Fallback local siempre activo por redundancia y seguridad
        self.inicializado = False
        
        # Ruta de las credenciales
        key_path = os.path.join(config.BASE_DIR, config.DATABASE["firebase_key_path"])
        db_url = config.DATABASE["firebase_db_url"]
        
        if not os.path.exists(key_path):
            print(f"[FIREBASE WARNING] No se encontro el archivo de credenciales '{os.path.basename(key_path)}'.")
            print("[FIREBASE WARNING] Se utilizara únicamente la Base de Datos SQLite Local.")
            return
            
        try:
            import firebase_admin
            from firebase_admin import credentials, db
            
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': db_url
            })
            self.db_reference = db.reference('/')
            self.inicializado = True
            print(f"[FIREBASE OK] Conectado exitosamente a Firebase RTDB: {db_url}")
        except Exception as e:
            print(f"[FIREBASE ERROR] No se pudo inicializar la conexion: {e}")
            print("[FIREBASE ERROR] Sincronizacion desactivada (Fallback a SQLite Local).")
            
    def registrar_acceso(self, camara_id: str, persona: str, confianza: float):
        # Registrar localmente primero (redundancia offline)
        self.local_db.registrar_acceso(camara_id, persona, confianza)
        
        if self.inicializado:
            try:
                from firebase_admin import db
                ref_acceso = db.reference('/accesos').push()
                ref_acceso.set({
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "camara_id": camara_id,
                    "persona": persona,
                    "confianza": float(confianza)
                })
                print(f"[FIREBASE CLOUD] Sincronizado acceso en la nube para: {persona}")
            except Exception as e:
                print(f"[FIREBASE CLOUD ERROR] Error al sincronizar acceso: {e}")
                
    def registrar_alerta(self, camara_id: str, tipo_alerta: str, severidad: str, ruta_imagen: str = None):
        # Registrar localmente primero
        self.local_db.registrar_alerta(camara_id, tipo_alerta, severidad, ruta_imagen)
        
        if self.inicializado:
            try:
                from firebase_admin import db
                ref_alerta = db.reference('/alertas').push()
                
                # Omitir la ruta local de disco de la imagen, o subir el nombre del archivo
                nombre_imagen = os.path.basename(ruta_imagen) if ruta_imagen else None
                
                ref_alerta.set({
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "camara_id": camara_id,
                    "tipo_alerta": tipo_alerta,
                    "severidad": severidad,
                    "nombre_imagen": nombre_imagen
                })
                print(f"[FIREBASE CLOUD] [ALERTA] Sincronizada alerta en la nube: {tipo_alerta}")
            except Exception as e:
                print(f"[FIREBASE CLOUD ERROR] Error al sincronizar alerta: {e}")
                
    def cerrar(self):
        self.local_db.cerrar()

def obtener_base_datos() -> DatabaseAdapter:
    """Retorna la base de datos configurada en config.py."""
    db_type = config.DATABASE["type"].lower()
    if db_type == "firebase":
        return FirebaseDatabase()
    else:
        return SQLiteDatabase()
