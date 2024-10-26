from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
import numpy as np
import cv2
from configs.soccer import SoccerPitchConfiguration
from configs.drawing import draw_pitch, draw_points_on_pitch
from configs.view_transformer import ViewTransformer
from inference import get_model



# Ruta del modelo y del video
PLAYER_DETECTION_MODEL = YOLO("..\\fotball_map_pass\\models\\weights.onnx")
# Configuración del modelo de detección de cancha
PITCH_DETECTION_MODEL_ID = "pitch-c3e9w/5"
PITCH_DETECTION_MODEL = get_model(PITCH_DETECTION_MODEL_ID, "49fMB8oQq6GxbnlRsfVd")



SOURCE_VIDEO_PATH = "..\\fotball_map_pass\\videos\\video.mp4"
TARGET_VIDEO_PATH = "..\\fotball_map_pass\\videos\\video.mp4"


# Configuración de la cancha virtualizada
CONFIG = SoccerPitchConfiguration()


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

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info)


# Procesar el video frame por frame y mostrar en tiempo real
for _ in tqdm(range(total_frames)):
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

    # Extraer las coordenadas de los objetos detectados
    ball_detections = detections[detections.class_id == 0]  # Asumiendo que el ID 0 es el balón
    players_detections = detections[detections.class_id == 1]  # Asumiendo que el ID 1 es jugador

    # Obtener coordenadas de los centros de detección para el balón y los jugadores
    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    frame_players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    # Transformar estas coordenadas a la cancha 2D
    pitch_ball_xy = view_transformer.inverse_transform_points(frame_ball_xy)
    pitch_players_xy = view_transformer.inverse_transform_points(frame_players_xy)

    # Crear una copia del pitch para cada frame
    pitch_2d_frame = draw_pitch(CONFIG)

    # Dibujar el balón y los jugadores en la cancha 2D
    pitch_2d_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_ball_xy,
        face_color=sv.Color.WHITE,
        edge_color=sv.Color.BLACK,
        radius=10,
        pitch=pitch_2d_frame
    )

    pitch_2d_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_players_xy,
        face_color=sv.Color.from_hex("00BFFF"),  # Color para el equipo 1
        edge_color=sv.Color.BLACK,
        radius=10,
        pitch=pitch_2d_frame
    )

    # Mostrar el frame proyectado en tiempo real
    cv2.imshow("Cancha 2D con Detecciones", pitch_2d_frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
