# 🛡️ Guía de Inicio Rápido - Lector de Personas e IA de Seguridad

Este documento sirve como guía paso a paso para configurar, iniciar y operar el sistema de seguridad inteligente en cualquier sistema operativo (**Windows, macOS o Linux**).

---

## 🚀 Cómo iniciar el sistema con 1 solo clic

Hemos creado lanzadores automáticos para que no tengas que escribir comandos en la terminal cada vez que uses el programa:

### 🪟 En Windows
1. Abre la carpeta del proyecto en tu Explorador de Archivos.
2. Haz doble clic en el archivo **`Iniciar_Lector.bat`**.
3. *Opcional:* Puedes hacerle clic derecho al archivo ➡️ *Enviar a* ➡️ *Escritorio (crear acceso directo)*. En el escritorio, cámbiale el icono en propiedades por un escudo de seguridad para que parezca una aplicación nativa.

### 🍎 En macOS y 🐧 en Linux
1. Abre tu terminal dentro de la carpeta del proyecto.
2. Otorga permisos de ejecución al script (solo la primera vez):
   ```bash
   chmod +x iniciar_lector.sh
   ```
3. Ejecuta el script:
   ```bash
   ./iniciar_lector.sh
   ```

---

## 🛠️ Requisitos previos (Solo para nuevas computadoras)

Si vas a abrir el proyecto por primera vez en otra computadora (como tu laptop):
1. Asegúrate de tener **Python 3.10 o superior** instalado.
2. Abre la consola en la carpeta del proyecto y ejecuta el instalador inicial para crear el entorno virtual y descargar los modelos oficiales:
   ```powershell
   python scripts/setup_project.py
   ```

---

## 🧠 Guía de Operación de los Módulos de IA

### 1. Reconocimiento Facial (Con Auto-Aprendizaje Activo)
El reconocedor facial (**SFace**) está diseñado para aprender por sí mismo sobre la marcha para que no tengas que tomar fotos manuales de perfil de todos los empleados:
* **Foto Inicial:** Registra a una persona guardando **una sola foto de frente** en la carpeta `db_rostros/` (ej: `Nombre_Apellido.png`).
* **Cómo aprende perfiles:** La primera vez que esa persona mire a la cámara, el sistema la identificará de frente. A medida que camine y gire la cabeza, el rastreador de rostro la seguirá y **extraerá automáticamente sus perfiles en vivo**, guardándolos en el caché `db_rostros/embeddings.json`.
* **Caché:** Si quieres reiniciar la base de datos de firmas aprendidas, simplemente borra el archivo `db_rostros/embeddings.json` y el sistema volverá a calcular las firmas desde cero a partir de las imágenes de la carpeta.

### 2. Detección de Amenazas (Armas)
* **Modelo Utilizado:** Hemos restaurado el modelo pre-entrenado robusto de Hugging Face (`Subh775/Threat-Detection-YOLOv8n`), que es el más preciso y no genera falsas alarmas con tus manos o fondo.
* **Umbral de Sensibilidad:** Configurado por defecto a **`0.55`** para tener alta sensibilidad al detectar cuchillos/machetes de carnicero y armas de fuego en diferentes ángulos.
* **Clases dinámicas:** Si en el futuro entrenas un modelo propio en Google Colab con más fotos (versión 4 en adelante) y reemplazas el archivo `modelos/threat_detection.pt`, el sistema leerá los nombres de las clases automáticamente de tu nuevo archivo y las traducirá al español.
