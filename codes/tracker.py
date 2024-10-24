import supervision as sv
from ultralytics import YOLO
from configs.team import TeamClassifier

SOURCE_VIDEO_PATH = "..\\fotball_map_pass\\videos\\pov1_definitivo_og.mp4"
BALL_ID = 0
PLAYER_ID = 1
PLAYER_DETECTION_MODEL = YOLO("..\\fotball_map_pass\\models\\weights.onnx")

ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FFD700'),
    base=25,
    height=21,
    outline_thickness=1
)

tracker = sv.ByteTrack()
tracker.reset()

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)

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

ball_detections = detections[detections.class_id == BALL_ID]
ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

all_detections = detections[detections.class_id != BALL_ID]
all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
all_detections.class_id -= 1
all_detections = tracker.update_with_detections(detections=all_detections)

labels = [
    f"#{tracker_id}"
    for tracker_id
    in all_detections.tracker_id
]

annotated_frame = frame.copy()
annotated_frame = ellipse_annotator.annotate(
    scene=annotated_frame,
    detections=all_detections)
annotated_frame = label_annotator.annotate(
    scene=annotated_frame,
    detections=all_detections,
    labels=labels)
annotated_frame = triangle_annotator.annotate(
    scene=annotated_frame,
    detections=ball_detections)

sv.plot_image(annotated_frame)