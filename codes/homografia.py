from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
import numpy as np
import cv2
import pandas as pd
from configs.soccer import SoccerPitchConfiguration
from configs.drawing import draw_pitch, draw_points_on_pitch
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

# Configuración de anotadores
ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER
)

# Inicializar DataFrame para guardar los eventos de pase
pases_df = pd.DataFrame(columns=['emisor_id', 'receptor_id', 'frame'])

# Configurar el VideoCapture de OpenCV
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
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

# Variables para seguimiento de pases
jugador_emisor = None
jugador_receptor = None
pase_en_proceso = False
distancia_minima = 50  # Distancia mínima para considerar que la pelota salió de la zona del jugador
distancia_recepcion = 50  # Distancia máxima para considerar que un jugador recibe la pelota

def registrar_pase(jugador_emisor, jugador_receptor, frame_id):
    """Registra un pase en el DataFrame"""
    print(f"[Registro en DataFrame] Pase registrado de Jugador {jugador_emisor} a Jugador {jugador_receptor} en el frame {frame_id}")
    global pases_df
    pases_df = pd.concat([pases_df, pd.DataFrame([[jugador_emisor, jugador_receptor, frame_id]], columns=pases_df.columns)], ignore_index=True)

# Procesar el video frame por frame y mostrar en tiempo real
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
    
    # Anotaciones en el frame original
    annotated_frame = frame.copy()
    annotated_frame = ellipse_annotator.annotate(
        scene=annotated_frame,
        detections=all_detections
    )

    # Obtener coordenadas de los centros de detección para el balón y los jugadores
    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    frame_players_xy = all_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_positions = {id: xy for id, xy in zip(all_detections.tracker_id, frame_players_xy)}

    # Transformar estas coordenadas a la cancha 2D
    pitch_ball_xy = view_transformer.inverse_transform_points(frame_ball_xy)
    pitch_players_xy = view_transformer.inverse_transform_points(frame_players_xy)

    # Imprimir posiciones con ID
    #for tracker_id, (x, y) in zip(all_detections.tracker_id, pitch_players_xy):
        #print(f"Jugador {{ID: {tracker_id}}} posición x: ({x:.2f}), posición y: ({y:.2f})")

    # Detectar el pase en proceso
    if len(pitch_ball_xy) > 0:  # Si se detecta el balón
        ball_position = pitch_ball_xy[0]
        for player_id, player_position in players_positions.items():
            distancia = np.linalg.norm(np.array(ball_position) - np.array(player_position))

            # Detectar inicio de pase si la pelota sale del jugador emisor
            if jugador_emisor is not None and distancia > distancia_minima and not pase_en_proceso:
                pase_en_proceso = True  # Iniciar el proceso de pase
                print(f"[Inicio de Pase] Jugador {jugador_emisor} inició un pase en el frame {frame_id}")
            
            # Detectar jugador receptor cuando la pelota está cerca de otro jugador
            if pase_en_proceso and distancia <= distancia_recepcion:
                jugador_receptor = player_id
                print(f"[Recepción de Pase] Jugador {jugador_receptor} recibe el pase en el frame {frame_id}")
                registrar_pase(jugador_emisor, jugador_receptor, frame_id)
                pase_en_proceso = False  # Reiniciar el proceso para el próximo pase
                jugador_emisor = jugador_receptor
                jugador_receptor = None
                break  # Salir del bucle si se detectó el pase
    
    # Mostrar el frame proyectado en tiempo real
    cv2.imshow("Cancha 2D con Detecciones", draw_pitch(CONFIG))

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Guardar el DataFrame en un archivo Excel
pases_df.to_excel('pases_registrados.xlsx', index=False)

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
