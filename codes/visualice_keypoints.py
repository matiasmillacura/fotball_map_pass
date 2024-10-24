import numpy as np
import supervision as sv
from configs.soccer import SoccerPitchConfiguration
from configs.drawing import draw_pitch


from inference import get_model

PITCH_DETECTION_MODEL_ID = "pitch-c3e9w/5"
PITCH_DETECTION_MODEL = get_model(PITCH_DETECTION_MODEL_ID, "49fMB8oQq6GxbnlRsfVd")


SOURCE_VIDEO_PATH = "A:\\sports\\sports\\pov1_definitivo_og.mp4"
import cv2

class ViewTransformer:

  def __init__(self, source:np.ndarray, target: np.ndarray):
    source = source.astype(np.float32)
    target = target.astype(np.float32)
    self.m, _ = cv2.findHomography(source, target)

  def transform_points(self, points:np.ndarray) -> np.ndarray:
    points = points.reshape(-1, 1, 2).astype(np.float32)
    points = cv2.perspectiveTransform(points, self.m)
    return points.reshape(-1, 2).astype(np.float32)


CONFIG = SoccerPitchConfiguration()

annotated_frame = draw_pitch(CONFIG)

vertex_annotator = sv.VertexAnnotator(
    color = sv.Color.from_hex('#FF1493'),
    radius = 8
)

edge_annotator = sv.EdgeAnnotator(
    color=sv.Color.from_hex('00BFFF'),
    thickness=2,
    edges=CONFIG.edges
)

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)

result = PITCH_DETECTION_MODEL.infer(frame, confidence=0.9)[0]
key_points = sv.KeyPoints.from_inference(result)
filter = key_points.confidence[0] > 0.95
frame_reference_points = key_points.xy[0][filter]
frame_reference_keypoints = sv.KeyPoints(xy=frame_reference_points[np.newaxis, ...])
pitch_reference_points = np.array(CONFIG.vertices)[filter]



view_transformer = ViewTransformer(
    source = pitch_reference_points,
    target = frame_reference_points
)

pitch_all_points = np.array(CONFIG.vertices)
frame_all_points = view_transformer.transform_points(pitch_all_points)
frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])

annotated_frame = frame.copy()
annotated_frame = edge_annotator.annotate(annotated_frame, frame_all_key_points)
annotated_frame = vertex_annotator.annotate(annotated_frame, frame_reference_keypoints)

sv.plot_image(annotated_frame)

