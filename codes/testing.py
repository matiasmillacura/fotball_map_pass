from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
import numpy as np
import cv2
import pandas as pd
from configs.soccer import SoccerPitchConfiguration
from configs.view_transformer import ViewTransformer
from inference import get_model

# Ruta del modelo y del video
PLAYER_DETECTION_MODEL = YOLO("..\\fotball_map_pass\\models\\weights.onnx")
PITCH_DETECTION_MODEL_ID = "pitch-c3e9w/5"
PITCH_DETECTION_MODEL = get_model(PITCH_DETECTION_MODEL_ID, "49fMB8oQq6GxbnlRsfVd")

SOURCE_VIDEO_PATH = "..\\fotball_map_pass\\videos\\corte pov 2.mp4"
CONFIG = SoccerPitchConfiguration()

# Configuración de tracker
tracker = sv.ByteTrack()
tracker.reset()

# Inicializar DataFrame para guardar las posiciones
posiciones_df = pd.DataFrame(columns=['Id', 'Pos X', 'Pos Y', 'Ball X', 'Ball Y'])

# Configurar el VideoCapture de OpenCV
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Detección de puntos clave en la imagen del video para la homografía
ret, frame = cap.read()
if not ret:
    print("No se pudo leer el frame inicial del video.")
    cap.release()
    exit()

result = PITCH_DETECTION_MODEL.infer(frame, confidence=0.9)[0]
key_points = sv.KeyPoints.from_inference(result)
filter = key_points.confidence[0] > 0.95
frame_reference_points = key_points.xy[0][filter]
pitch_reference_points = np.array(CONFIG.vertices)[filter]

# Crear el objeto ViewTransformer
view_transformer = ViewTransformer(
    source=pitch_reference_points,
    target=frame_reference_points
)

# Variables para seguimiento de pases y posiciones del balón
ultima_posicion_balon = None  # Última posición conocida del balón

# Procesar el video frame por frame
for frame_id in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Realizar la inferencia
    detections_result = PLAYER_DETECTION_MODEL.predict(frame, imgsz=1792)[0]
    boxes = detections_result.boxes.xyxy.cpu().numpy()  # Coordenadas de las cajas
    scores = detections_result.boxes.conf.cpu().numpy()  # Puntajes de confianza
    class_ids = detections_result.boxes.cls.cpu().numpy().astype(int)  # IDs de clases

    # Convertir las detecciones a un formato compatible
    detections = sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=class_ids
    )

    # Filtrar el balón y jugadores
    ball_detections = detections[detections.class_id == 0]  # Asumiendo que el ID 0 es el balón
    players_detections = detections[detections.class_id == 1]  # Asumiendo que el ID 1 es jugador

    # Aplicar el tracker a las detecciones de jugadores
    all_detections = tracker.update_with_detections(detections=players_detections)
    
    # Obtener coordenadas de los centros de detección para el balón y los jugadores
    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    frame_players_xy = all_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_positions = {id: xy for id, xy in zip(all_detections.tracker_id, frame_players_xy)}

    # Usar última posición conocida del balón si no hay detección en este frame
    if frame_ball_xy is not None and len(frame_ball_xy) > 0:  # Si se detecta el balón en este frame
        ultima_posicion_balon = frame_ball_xy[0]  # Actualizar la última posición conocida
    elif ultima_posicion_balon is not None:
        frame_ball_xy = [ultima_posicion_balon]  # Usar última posición conocida como sustituto

    # Asegurar que frame_ball_xy es un array de numpy
    frame_ball_xy = np.array(frame_ball_xy)

    # Transformar estas coordenadas a la cancha 2D
    if frame_ball_xy.size > 0:
        pitch_ball_xy = view_transformer.inverse_transform_points(frame_ball_xy)[0]  # Obtener como punto único
    else:
        pitch_ball_xy = ultima_posicion_balon  # Usar la última posición conocida del balón

    # Guardar las posiciones en el DataFrame
    for player_id, player_position in players_positions.items():
        pitch_player_xy = view_transformer.inverse_transform_points(np.array([player_position]))[0]
        posiciones_df = pd.concat([posiciones_df, pd.DataFrame([[player_id, pitch_player_xy[0], pitch_player_xy[1], pitch_ball_xy[0], pitch_ball_xy[1]]], 
                                                              columns=posiciones_df.columns)], ignore_index=True)

# Guardar el DataFrame en un archivo Excel
posiciones_df.to_excel('posiciones_jugadores_balon.xlsx', index=False)

# Liberar recursos
cap.release()
