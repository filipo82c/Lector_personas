import os
import sys
import cv2

# Agregar src/ al PATH para poder usar config y detector
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import config

def main():
    print("=================================================================")
    print("               REGISTRADOR DE ROSTROS - LECTOR                   ")
    print("=================================================================")
    
    # Pedir nombre de la persona
    nombre_entrada = input("\nIngrese el NOMBRE y APELLIDO del usuario a registrar: ").strip()
    if not nombre_entrada:
        print("[ERROR] Nombre invalido.")
        sys.exit(1)
        
    nombre_formateado = nombre_entrada.replace(" ", "_")
    img_dest_path = os.path.join(config.DB_DIR, f"{nombre_formateado}.jpg")
    
    # 1. Cargar el detector de rostros para validación en caliente
    print("\nInicializando detector de rostros YuNet...")
    yunet_path = config.DETECTOR["yunet_path"]
    det_threshold = config.DETECTOR["face_det_threshold"]
    
    try:
        detector_face = cv2.FaceDetectorYN.create(
            yunet_path, "", (320, 320), det_threshold, 0.3, 5000,
            cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
        )
    except Exception as e:
        print(f"[ERROR] No se pudo cargar YuNet: {e}")
        sys.exit(1)

    # 2. Iniciar camara
    print("\nAbriendo camara... Por favor posicione su rostro frente a ella.")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] No se pudo acceder a la camara web.")
        sys.exit(1)
        
    print("\nINSTRUCCIONES:")
    print(" - Presione [ESPACIO] para capturar el rostro.")
    print(" - Presione [Q] para cancelar y salir.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Fallo al leer de la camara.")
            break
            
        h, w = frame.shape[:2]
        
        # Copia para pintar una guia de deteccion
        preview_frame = frame.copy()
        detector_face.setInputSize((w, h))
        ret_det, faces = detector_face.detect(frame)
        
        face_detected = False
        if ret_det and faces is not None:
            face_detected = True
            # Dibujar un recuadro verde si hay rostro detectado
            for face in faces:
                fx, fy, fw, fh = map(int, face[0:4])
                cv2.rectangle(preview_frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                cv2.putText(preview_frame, "Rostro Listo para Capturar", (fx, fy - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # Guia de advertencia si no detecta rostro
            cv2.putText(preview_frame, "No se detecta rostro. Acerquese o ilumine su cara.", 
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
        cv2.imshow("Registro Facial - Vista Previa", preview_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '): # Espacio para capturar
            if face_detected:
                # Guardar la foto
                cv2.imwrite(img_dest_path, frame)
                print(f"\n[OK] Foto capturada y guardada en: {os.path.relpath(img_dest_path, config.BASE_DIR)}")
                
                # Intentar borrar el cache JSON de embeddings para que se regenere al arrancar
                embeddings_json_path = os.path.join(config.DB_DIR, "embeddings.json")
                if os.path.exists(embeddings_json_path):
                    try:
                        os.remove(embeddings_json_path)
                        print("[OK] Cache de base de datos facial actualizado.")
                    except Exception as e:
                        print(f"[!] No se pudo borrar el cache embeddings.json: {e}")
                break
            else:
                print("\n[!] No se puede capturar: No se ha detectado ningun rostro en el cuadro actual.")
                
        elif key == ord('q') or key == 27: # 'q' o Esc para salir
            print("\n[INFO] Registro cancelado por el usuario.")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("\n=== Proceso de Registro Finalizado ===")

if __name__ == "__main__":
    main()
