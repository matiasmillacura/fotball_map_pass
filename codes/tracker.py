import cv2
import numpy as np
from pathlib import Path
from boxmot import BoTSORT
from ultralytics import YOLO

# Ajustar la dimensión de la imagen antes de pasarla al modelo
expected_width = 1792
expected_height = 1792

# Ruta al modelo YOLO entrenado (puedes cambiarlo por otro modelo si lo deseas)
yolo_model_path = '..\\fotball_map_pass\\models\\weights.onnx'
yolo_model = YOLO(yolo_model_path)

# Cargar el modelo de ReID
reid_model_path = Path('resnet50_fc512_msmt17.pt')

# Inicializar el tracker BotSORT con ReID
tracker = BoTSORT(
    reid_weights=reid_model_path,  # Modelo de ReID
    device='cpu',                  # Usa 'cuda' si tienes GPU, 'cpu' si no
    half=False,                     # Usa 'half' si deseas operaciones en FP16
    track_low_thresh=0.4,
    new_track_thresh=0.8,
    track_buffer=1000,              # Aumentar el tiempo de vida de las pistas
    match_thresh=0.8,             # Umbral de coincidencia en la asociación
    proximity_thresh=0.7,         # Tolerancia en IoU para la primera asociación
    appearance_thresh=0.3,        # Umbral de coincidencia de características visuales
    frame_rate=59.94,                # Tasa de cuadros del video
    fuse_first_associate=True,    # Fusionar apariencia y movimiento en la primera asociación
    with_reid=True                # Activar ReID para evitar pérdida de IDs       
)

# Ruta al video que deseas procesar
video_path = '..\\fotball_map_pass\\videos\\corte pov 2.mp4'
cap = cv2.VideoCapture(video_path)

# Abrir el archivo de texto para guardar las detecciones y seguimientos
with open('detecciones_y_seguimiento.txt', 'w') as f:
    frame_count = 0  # Contador de fotogramas

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar la imagen al tamaño esperado por el modelo
        resized_frame = cv2.resize(frame, (expected_width, expected_height))

        # Realizar detecciones con YOLOv8
        results = yolo_model.predict(source=resized_frame, imgsz=(expected_width, expected_height))

        # Procesar detecciones para el tracker BotSORT
        detections = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()  # Coordenadas [x1, y1, x2, y2]
                conf = box.conf[0].cpu().numpy()  # Confianza
                cls = int(box.cls[0].cpu().numpy())  # Clase (jugador o balón)
                detections.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls])

        if len(detections) > 0:
            # Convertir las detecciones en un formato compatible con BoTSORT
            dets = np.array(detections)

            # Actualizar el tracker con las detecciones
            tracks = tracker.update(dets, resized_frame)

            # Escribir las detecciones y seguimientos en el archivo de texto
            f.write(f"Frame {frame_count}:\n")
            for track in tracks:
                track_id = track[4]  # ID del objeto
                x1, y1, x2, y2 = track[:4]
                class_name = "JUGADOR" if int(track[5]) == 1 else "BALON"
                f.write(f"  ID: {track_id}, Clase: {class_name}, Coordenadas: ({x1}, {y1}, {x2}, {y2})\n")

                # Dibujar las cajas de seguimiento y sus IDs
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Mostrar el video con el seguimiento
        cv2.imshow('Tracking con BotSORT', frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Incrementar el contador de fotogramas
        frame_count += 1

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
