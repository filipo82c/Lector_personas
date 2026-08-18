# 🕵️ Lector de Personas IA - Reconocimiento Facial y Detección de Amenazas

Sistema de vigilancia inteligente optimizado para sectores privados y entornos de alta seguridad. Diseñado para procesar múltiples flujos de video simultáneamente (cámaras de seguridad RTSP o cámaras web locales) en segundo plano, reconociendo empleados/estudiantes a distancia mediante Deep Learning local y detectando amenazas crítias (armas de fuego y blancas) de forma instantánea.

---

## 🌟 Características Principales

*   **Detección y Reconocimiento Facial de Largo Alcance:** Utiliza **YuNet** (detector facial de alta velocidad) y **SFace** (extractor de firmas de 128 dimensiones) a través de OpenCV DNN, logrando capturas de rostros a varios metros de distancia de forma fluida.
*   **Aceleración de Hardware AMD / Ryzen CPU:** Optimizado para correr a máxima velocidad en procesadores multinúcleo Ryzen y GPUs AMD usando aceleración gráfica nativa **OpenCL** en OpenCV.
*   **Detección de Amenazas en Tiempo Real:** Incorpora el modelo **YOLO11** (Nano) capaz de identificar instantáneamente armas de fuego, armas blancas (cuchillos) y personas completas, activando protocolos de alerta.
*   **Procesamiento Multicámara Concurrente:** Arquitectura multihilo (`CameraWorker`). Cada cámara corre en su propio subproceso de captura e inferencia para evitar retrasos y aprovechar la CPU de 8 núcleos / 16 hilos.
*   **Persistencia Modular y Sincronización en Tiempo Real:**
    *   **Local (SQLite):** Registra accesos y alertas en una base de datos local `seguridad.db`.
    *   **Nube (Firebase Realtime Database):** Sincroniza datos en tiempo real al conectarlo con tu cuenta de Firebase.
*   **Diseñado para Distribución Comercial (Doble Clic):** Viene listo con un script para empaquetar toda la aplicación y sus modelos en una carpeta auto-contenida con un ejecutable `.exe` de Windows para fácil instalación en empresas cliente.

---

## 🏗️ Estructura del Proyecto

*   `modelos/`: Contiene los modelos `.onnx` de YuNet y SFace, y las pesas `.pt` de YOLO11.
*   `db_rostros/`: Almacena las fotos de los empleados (`Nombre_Apellido.jpg`) y el archivo caché indexado `embeddings.json` para carga inmediata.
*   `alertas/`: Guarda automáticamente capturas de pantalla de evidencias de amenazas o intrusos detectados.
*   `src/`: Código fuente principal (Configuración, Base de Datos, Captura de Video, Inferencia de IA e Hilo principal).
*   `scripts/`: Scripts automatizados para la inicialización (`setup_project.py`), registro facial interactivo (`register_face.py`) y compilación (`compile_project.py`).

---

## 🚀 Instalación y Configuración Rápida

### 1. Clonar el repositorio y configurar entorno
Asegúrate de estar en el directorio raíz del proyecto y ejecuta:
```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Descargar Modelos y Estructurar Carpetas
Ejecuta el script automatizado que creará los directorios y descargará todos los archivos de redes neuronales (YuNet, SFace, YOLO11, Threat Model):
```bash
python scripts/setup_project.py
```

### 3. Registrar Rostros de Forma Interactiva
Para registrar un nuevo empleado o compañero de clase, corre:
```bash
python scripts/register_face.py
```
*Ingresa el Nombre y Apellido de la persona, posiciónate frente a la cámara y presiona **[ESPACIO]** cuando veas el recuadro verde para guardar su foto. Presiona **[Q]** para salir.*

### 4. Configurar Cámaras y Base de Datos (`config.json`)
El archivo `config.json` se creará automáticamente en la raíz en la primera ejecución. Edítalo para agregar tus cámaras y configuración de base de datos:

```json
{
    "camaras": [
        {"id": "Camara_Web_Principal", "source": 0},
        {"id": "Camara_Pasillo_RTSP", "source": "rtsp://usuario:contraseña@192.168.1.100:554/stream1"}
    ],
    "database": {
        "type": "sqlite", // Cambia a "firebase" si deseas sincronización en la nube
        "sqlite_db_name": "seguridad.db",
        "firebase_db_url": "https://tu-proyecto.firebaseio.com/",
        "firebase_key_path": "firebase_key.json"
    }
}
```

### 5. Iniciar la Vigilancia en Tiempo Real
Inicia el sistema central multihilo:
```bash
python src/main.py
```
*Se abrirán ventanas en tiempo real para cada una de las cámaras. Presiona **[Q]** en cualquiera de ellas para apagar el sistema limpiamente.*

---

## 📦 Empaquetado para Venta / Despliegue en Clientes (`.exe`)

Para entregar el software a una empresa cliente sin requerir que instalen Python, ejecuta el compilador automático:
```bash
python scripts/compile_project.py
```
Al finalizar, tendrás una carpeta lista para distribución en `dist/LectorPersonasIA/`. Solo necesitas copiar esa carpeta en la computadora del cliente, editar su `config.json` interno con sus cámaras y base de datos, y hacer doble clic en `LectorPersonasIA.exe`.
