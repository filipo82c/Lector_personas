import os
import sys
import time
import datetime
import threading
import tkinter as tk
from tkinter import ttk, messagebox
try:
    from PIL import Image, ImageTk, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import engine

class SecurityAppGUI:
    """Interfaz de Escritorio de Nivel Empresarial para Centro de Control de Seguridad (SOC)."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("SISTEMA CENTRAL DE VIGILANCIA IA - CENTRO DE CONTROL")
        
        # Adaptabilidad Multi-Monitor (Detectar resolución de la pantalla actual)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        
        w = max(800, int(sw * 0.90))
        h = max(500, int(sh * 0.85))
        
        self.root.geometry(f"{w}x{h}")
        self.root.minsize(760, 460) # Mínimo flexible para pantallas pequeñas de laptop
        
        # Intentar maximizar de forma nativa en la pantalla actual
        try:
            self.root.attributes('-zoomed', True)
        except Exception:
            try:
                self.root.state('zoomed')
            except Exception:
                pass
                
        self.is_fullscreen = False
        self.root.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.root.bind("<Escape>", lambda e: self.exit_fullscreen())
        
        # Paleta de Colores Corporativa Profesional (Executive Obsidian Dark Mode)
        self.colors = {
            "bg_main": "#0b0e14",       # Obsidian Dark background
            "panel_bg": "#141923",      # Deep Navy Panel
            "panel_border": "#1e2638",  # Subtle Border
            "header_bg": "#10141e",     # Top Header Dark
            "text_bright": "#ffffff",   # Pure White
            "text_muted": "#8a99ad",    # Cool Grey Muted
            "accent_green": "#10b981",  # Emerald Active
            "accent_green_hover": "#059669",
            "accent_red": "#e11d48",    # Rose Red Stop
            "accent_red_hover": "#be123c",
            "accent_blue": "#2563eb",   # Sapphire Accent
            "accent_amber": "#f59e0b",  # Amber Warning
            "table_bg": "#0f131c",      # Darker Table Shading
            "table_alt": "#161c28"      # Alternate Row
        }
        
        self.root.configure(bg=self.colors["bg_main"])
        
        # Inicializar Motor de IA
        self.security_engine = engine.SecurityEngine()
        self.security_engine.register_alert_callback(self.on_threat_alert)
        self.security_engine.register_access_callback(self.on_access_event)
        
        # Cargar Ícono de Aplicación
        icon_path = os.path.join(config.BASE_DIR, "assets", "icon.png")
        if not os.path.exists(icon_path):
            icon_path = os.path.join(config.BASE_DIR, "icon.png")
        if os.path.exists(icon_path):
            try:
                if HAS_PIL:
                    icon_img = ImageTk.PhotoImage(Image.open(icon_path))
                    self.root.iconphoto(True, icon_img)
            except Exception as e:
                print(f"[!] Aviso al cargar icono: {e}")
                
        # Variables de control
        self.current_camera_id = config.CAMARAS[0]["id"] if config.CAMARAS else "Camara_Web_0"
        self.camera_source_var = tk.StringVar(value=str(config.CAMARAS[0]["source"]) if config.CAMARAS else "0")
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 30
        
        # Configurar Estilos ttk
        self.setup_styles()
        
        # Construir Interfaz Gráfica Empresarial
        self.setup_ui()
        
        # Bucle de actualización de cuadros de video (30 FPS)
        self.update_video_loop()
        
        # Cierre seguro
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('default')
        
        # Estilo para Pestañas
        style.configure('TNotebook', background=self.colors["panel_bg"], borderwidth=0)
        style.configure('TNotebook.Tab', 
                        background=self.colors["bg_main"], 
                        foreground=self.colors["text_muted"], 
                        padding=[16, 8], 
                        font=("Helvetica", 9, "bold"), 
                        borderwidth=0)
        style.map('TNotebook.Tab', 
                  background=[('selected', self.colors["accent_blue"])], 
                  foreground=[('selected', self.colors["text_bright"])])
        
        # Estilo para Tablas (Treeview)
        style.configure('Treeview', 
                        background=self.colors["table_bg"], 
                        foreground=self.colors["text_bright"], 
                        fieldbackground=self.colors["table_bg"], 
                        rowheight=26, 
                        font=("Helvetica", 9),
                        borderwidth=0)
        style.configure('Treeview.Heading', 
                        background=self.colors["panel_bg"], 
                        foreground=self.colors["text_muted"], 
                        font=("Helvetica", 8, "bold"), 
                        relief="flat")
        style.map('Treeview', background=[('selected', self.colors["panel_border"])])

    def setup_ui(self):
        # =========================================================================
        # 1. HEADER BAR CORPORATIVO
        # =========================================================================
        header_frame = tk.Frame(self.root, bg=self.colors["header_bg"], height=55, bd=0)
        header_frame.pack(fill="x", side="top")
        
        # Logo / Título Principal
        title_box = tk.Frame(header_frame, bg=self.colors["header_bg"])
        title_box.pack(side="left", padx=20, pady=10)
        
        main_title = tk.Label(
            title_box, 
            text="LECTOR DE PERSONAS IA", 
            font=("Helvetica", 14, "bold"), 
            fg=self.colors["text_bright"], 
            bg=self.colors["header_bg"]
        )
        main_title.pack(side="left")
        
        sub_title = tk.Label(
            title_box, 
            text="  |  CENTRO DE CONTROL Y MONITOREO DE SEGURIDAD", 
            font=("Helvetica", 9, "bold"), 
            fg=self.colors["text_muted"], 
            bg=self.colors["header_bg"]
        )
        sub_title.pack(side="left")
        
        # Status Pill Indicator
        self.status_pill = tk.Label(
            header_frame, 
            text="SISTEMA EN PAUSA", 
            font=("Helvetica", 9, "bold"), 
            fg=self.colors["text_bright"], 
            bg=self.colors["accent_amber"], 
            padx=14, 
            pady=4
        )
        self.status_pill.pack(side="right", padx=20, pady=12)

        # Botón Pantalla Completa / Maximizar
        btn_fullscreen = tk.Button(
            header_frame, 
            text="MAXIMIZAR [F11]", 
            font=("Helvetica", 8, "bold"), 
            fg=self.colors["text_bright"], 
            bg=self.colors["panel_border"], 
            activebackground=self.colors["accent_blue"], 
            activeforeground="#ffffff",
            bd=0, 
            padx=10, 
            pady=3, 
            cursor="hand2", 
            relief="flat",
            command=self.toggle_fullscreen
        )
        btn_fullscreen.pack(side="right", padx=(0, 5), pady=12)
        
        # Linea divisora decorativa
        divider = tk.Frame(self.root, bg=self.colors["panel_border"], height=1)
        divider.pack(fill="x", side="top")

        # =========================================================================
        # 2. BARRA DE CONTROL Y PARÁMETROS
        # =========================================================================
        control_frame = tk.Frame(self.root, bg=self.colors["panel_bg"], bd=0)
        control_frame.pack(fill="x", side="top", padx=15, pady=12)
        
        # Botón Iniciar
        self.btn_start = tk.Button(
            control_frame, 
            text="INICIAR VIGILANCIA", 
            font=("Helvetica", 10, "bold"), 
            fg=self.colors["text_bright"], 
            bg=self.colors["accent_green"], 
            activebackground=self.colors["accent_green_hover"], 
            activeforeground=self.colors["text_bright"],
            bd=0, 
            padx=22, 
            pady=9, 
            cursor="hand2",
            relief="flat",
            command=self.start_surveillance
        )
        self.btn_start.pack(side="left", padx=(10, 8), pady=8)
        
        # Botón Detener
        self.btn_stop = tk.Button(
            control_frame, 
            text="DETENER VIGILANCIA", 
            font=("Helvetica", 10, "bold"), 
            fg=self.colors["text_bright"], 
            bg=self.colors["accent_red"], 
            activebackground=self.colors["accent_red_hover"], 
            activeforeground=self.colors["text_bright"],
            bd=0, 
            padx=22, 
            pady=9, 
            cursor="hand2",
            relief="flat",
            state="disabled",
            command=self.stop_surveillance
        )
        self.btn_stop.pack(side="left", padx=8, pady=8)
        
        # Separador Vertical
        v_sep = tk.Frame(control_frame, bg=self.colors["panel_border"], width=1, height=30)
        v_sep.pack(side="left", padx=20)
        
        # Fuente de Video
        cam_label = tk.Label(
            control_frame, 
            text="FUENTE DE CÁMARA:", 
            font=("Helvetica", 8, "bold"), 
            fg=self.colors["text_muted"], 
            bg=self.colors["panel_bg"]
        )
        cam_label.pack(side="left", padx=(0, 8))
        
        self.cam_entry = tk.Entry(
            control_frame, 
            textvariable=self.camera_source_var, 
            font=("Consolas", 10), 
            bg=self.colors["bg_main"], 
            fg=self.colors["text_bright"], 
            insertbackground=self.colors["text_bright"], 
            relief="flat",
            width=18, 
            bd=5
        )
        self.cam_entry.pack(side="left", padx=5)
        
        # Separador Vertical
        v_sep2 = tk.Frame(control_frame, bg=self.colors["panel_border"], width=1, height=30)
        v_sep2.pack(side="left", padx=20)
        
        # Control Deslizante de Sensibilidad
        sens_label = tk.Label(
            control_frame, 
            text="SENSIBILIDAD AMENAZAS:", 
            font=("Helvetica", 8, "bold"), 
            fg=self.colors["text_muted"], 
            bg=self.colors["panel_bg"]
        )
        sens_label.pack(side="left", padx=(0, 8))
        
        self.sens_scale = tk.Scale(
            control_frame, 
            from_=0.50, 
            to=0.99, 
            resolution=0.01, 
            orient="horizontal", 
            bg=self.colors["panel_bg"], 
            fg=self.colors["text_bright"], 
            highlightthickness=0, 
            troughcolor=self.colors["bg_main"], 
            activebackground=self.colors["accent_blue"],
            length=160,
            command=self.on_sens_change
        )
        self.sens_scale.set(config.DETECTOR.get("yolo_threat_threshold", 0.90))
        self.sens_scale.pack(side="left", padx=5)

        # =========================================================================
        # 3. CONTENIDO PRINCIPAL (VIDEO & TABLAS)
        # =========================================================================
        content_frame = tk.Frame(self.root, bg=self.colors["bg_main"])
        content_frame.pack(fill="both", expand=True, padx=15, pady=(5, 10))
        
        content_frame.grid_columnconfigure(0, weight=3)
        content_frame.grid_columnconfigure(1, weight=1, minsize=420)
        content_frame.grid_rowconfigure(0, weight=1)
        
        # PANEL IZQUIERDO: VISOR DE VIDEO
        video_panel = tk.Frame(content_frame, bg=self.colors["panel_bg"], bd=1, highlightbackground=self.colors["panel_border"], highlightthickness=1)
        video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        
        video_top_bar = tk.Frame(video_panel, bg=self.colors["panel_bg"], height=35)
        video_top_bar.pack(fill="x", side="top", padx=12, pady=8)
        
        video_title = tk.Label(
            video_top_bar, 
            text="TRANSMISIÓN DE VIDEO EN TIEMPO REAL", 
            font=("Helvetica", 9, "bold"), 
            fg=self.colors["text_muted"], 
            bg=self.colors["panel_bg"]
        )
        video_title.pack(side="left")
        
        self.resolution_badge = tk.Label(
            video_top_bar, 
            text="HD 1080p | 30 FPS", 
            font=("Helvetica", 8, "bold"), 
            fg=self.colors["accent_blue"], 
            bg=self.colors["panel_bg"]
        )
        self.resolution_badge.pack(side="right")
        
        # Canvas de Renderizado de Video
        self.video_canvas = tk.Label(video_panel, bg="#05070a")
        self.video_canvas.pack(fill="both", expand=True, padx=8, pady=5)
        self.video_canvas.bind("<Configure>", lambda e: self.on_canvas_resize())
        
        # Renderizar gráfico de espera inicial
        self.render_placeholder_graphic()
        
        # Footer del Panel de Video
        self.fps_label = tk.Label(
            video_panel, 
            text="ESTADO: Vigilancia pausada. Presione 'INICIAR VIGILANCIA' para activar el motor de IA.", 
            font=("Helvetica", 8), 
            fg=self.colors["text_muted"], 
            bg=self.colors["panel_bg"]
        )
        self.fps_label.pack(side="bottom", anchor="w", padx=12, pady=8)

        # PANEL DERECHO: PESTAÑAS DE REGISTROS Y ALERTAS
        logs_panel = tk.Frame(content_frame, bg=self.colors["panel_bg"], bd=1, highlightbackground=self.colors["panel_border"], highlightthickness=1)
        logs_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        
        logs_header = tk.Label(
            logs_panel, 
            text="REGISTRO DE EVENTOS Y DETECCIONES", 
            font=("Helvetica", 9, "bold"), 
            fg=self.colors["text_bright"], 
            bg=self.colors["panel_bg"]
        )
        logs_header.pack(side="top", anchor="w", padx=12, pady=10)
        
        self.notebook = ttk.Notebook(logs_panel)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=5)
        
        # Pestaña 1: Amenazas (Armas)
        self.tab_threats = tk.Frame(self.notebook, bg=self.colors["table_bg"])
        self.notebook.add(self.tab_threats, text="Amenazas Detectadas")
        
        columns_threats = ("hora", "tipo", "confianza", "nivel")
        self.tree_threats = ttk.Treeview(self.tab_threats, columns=columns_threats, show="headings", selectmode="browse")
        self.tree_threats.heading("hora", text="HORA")
        self.tree_threats.heading("tipo", text="AMENAZA DETECTADA")
        self.tree_threats.heading("confianza", text="CONFIANZA")
        self.tree_threats.heading("nivel", text="NIVEL")
        
        self.tree_threats.column("hora", width=70, anchor="center")
        self.tree_threats.column("tipo", width=180, anchor="w")
        self.tree_threats.column("confianza", width=80, anchor="center")
        self.tree_threats.column("nivel", width=80, anchor="center")
        
        self.tree_threats.pack(fill="both", expand=True)

        # Pestaña 2: Personas (Rostros)
        self.tab_faces = tk.Frame(self.notebook, bg=self.colors["table_bg"])
        self.notebook.add(self.tab_faces, text="Registro de Personas")
        
        columns_faces = ("hora", "persona", "confianza", "estado")
        self.tree_faces = ttk.Treeview(self.tab_faces, columns=columns_faces, show="headings", selectmode="browse")
        self.tree_faces.heading("hora", text="HORA")
        self.tree_faces.heading("persona", text="NOMBRE Y APELLIDO")
        self.tree_faces.heading("confianza", text="MATCH")
        self.tree_faces.heading("estado", text="ESTADO")
        
        self.tree_faces.column("hora", width=70, anchor="center")
        self.tree_faces.column("persona", width=180, anchor="w")
        self.tree_faces.column("confianza", width=80, anchor="center")
        self.tree_faces.column("estado", width=80, anchor="center")
        
        self.tree_faces.pack(fill="both", expand=True)

        # =========================================================================
        # 4. FOOTER GENERAL
        # =========================================================================
        footer_frame = tk.Frame(self.root, bg=self.colors["header_bg"], height=28)
        footer_frame.pack(fill="x", side="bottom")
        
        self.footer_label = tk.Label(
            footer_frame, 
            text=f"Base de Datos: SQLite | Identidades Registradas: {len(self.security_engine.detector_ia.db_embeddings)} | Arquitectura Multihilo Activa", 
            font=("Helvetica", 8), 
            fg=self.colors["text_muted"], 
            bg=self.colors["header_bg"]
        )
        self.footer_label.pack(side="left", padx=15, pady=4)

    def render_placeholder_graphic(self):
        """Genera un gráfico vectorial limpio en el canvas cuando la cámara no está transmitiendo."""
        import numpy as np
        w = max(400, self.video_canvas.winfo_width())
        h = max(300, self.video_canvas.winfo_height())
        img = np.zeros((h, w, 3), dtype=np.uint8)
            
        # Dibujar líneas de retícula tecnológica
        color_grid = (25, 32, 45)
        for x in range(0, w, 40):
            cv2.line(img, (x, 0), (x, h), color_grid, 1)
        for y in range(0, h, 40):
            cv2.line(img, (0, y), (w, y), color_grid, 1)
            
        # Círculo central de radar
        cx, cy = w // 2, h // 2
        cv2.circle(img, (cx, cy), 75, (35, 45, 65), 2)
        cv2.circle(img, (cx, cy), 130, (30, 40, 58), 1)
        
        # Texto central profesional
        cv2.putText(img, "SISTEMA EN PAUSA", (cx - 110, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (138, 153, 173), 2, cv2.LINE_AA)
        cv2.putText(img, "Haga clic en 'INICIAR VIGILANCIA' para activar el flujo", (cx - 190, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (85, 100, 120), 1, cv2.LINE_AA)
        
        # Convertir a formato Tkinter
        if HAS_PIL:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
        else:
            _, buffer = cv2.imencode('.ppm', img)
            img_tk = tk.PhotoImage(data=buffer.tobytes())
            
        self.video_canvas.config(image=img_tk)
        self.video_canvas.image = img_tk

    def toggle_fullscreen(self):
        """Alterna el modo Pantalla Completa en cualquier monitor."""
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)

    def exit_fullscreen(self):
        """Sale del modo Pantalla Completa."""
        if self.is_fullscreen:
            self.is_fullscreen = False
            self.root.attributes("-fullscreen", False)

    def on_canvas_resize(self):
        """Redibuja el gráfico vectorial cuando el usuario redimensiona o maximiza la ventana."""
        if not self.security_engine.is_running():
            self.render_placeholder_graphic()

    def start_surveillance(self):
        source_val = self.camera_source_var.get().strip()
        if source_val.isdigit():
            source_val = int(source_val)
            
        config.CAMARAS[0]["source"] = source_val
        
        if self.security_engine.start_surveillance():
            self.status_pill.config(text="VIGILANCIA ACTIVA", bg=self.colors["accent_green"])
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal")
            self.cam_entry.config(state="disabled")

    def stop_surveillance(self):
        if self.security_engine.stop_surveillance():
            self.status_pill.config(text="SISTEMA EN PAUSA", bg=self.colors["accent_amber"])
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.cam_entry.config(state="normal")
            self.render_placeholder_graphic()
            self.fps_label.config(text="ESTADO: Vigilancia pausada. Presione 'INICIAR VIGILANCIA' para activar el motor de IA.")

    def on_sens_change(self, val):
        val_float = float(val)
        self.security_engine.set_threat_threshold(val_float)

    def on_threat_alert(self, camera_id, event, frame):
        ahora = datetime.datetime.now().strftime("%H:%M:%S")
        detalle = event.get("detalle", "Amenaza").upper()
        conf = f"{int(event.get('confianza', 0) * 100)}%"
        nivel = "CRÍTICO"
        
        self.root.after(0, lambda: self.tree_threats.insert("", 0, values=(ahora, detalle, conf, nivel)))

    def on_access_event(self, camera_id, event, frame):
        ahora = datetime.datetime.now().strftime("%H:%M:%S")
        persona = event.get("persona", "Desconocido")
        conf = f"{int(event.get('confianza', 0) * 100)}%"
        estado = "AUTORIZADO" if persona != "Desconocido" else "INTRUSO"
        
        self.root.after(0, lambda: self.tree_faces.insert("", 0, values=(ahora, persona, conf, estado)))

    def update_video_loop(self):
        if self.security_engine.is_running():
            frame = self.security_engine.latest_frames.get(self.current_camera_id)
            if frame is not None:
                cw = max(320, self.video_canvas.winfo_width())
                ch = max(240, self.video_canvas.winfo_height())
                
                if HAS_PIL:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(frame_rgb)
                    img_pil.thumbnail((cw, ch), Image.Resampling.LANCZOS)
                    img_tk = ImageTk.PhotoImage(image=img_pil)
                else:
                    h, w = frame.shape[:2]
                    if w > cw or h > ch:
                        frame_resized = cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_AREA)
                    else:
                        frame_resized = frame
                    _, buffer = cv2.imencode('.ppm', frame_resized)
                    img_tk = tk.PhotoImage(data=buffer.tobytes())
                
                self.video_canvas.config(image=img_tk)
                self.video_canvas.image = img_tk
                
                self.fps_counter += 1
                if time.time() - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_time = time.time()
                    
                self.fps_label.config(text=f"ESTADO: Vigilancia en vivo | {self.current_fps} FPS | Cámara: {self.current_camera_id} | Inferencia Asíncrona Activa")
                
        self.root.after(30, self.update_video_loop)

    def on_close(self):
        if messagebox.askokcancel("Salir", "¿Desea cerrar el Centro de Control de Vigilancia?"):
            self.security_engine.cerrar()
            self.root.destroy()

def main():
    root = tk.Tk()
    app = SecurityAppGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
