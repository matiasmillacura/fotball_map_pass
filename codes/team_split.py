from tqdm import tqdm
import torch
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked
import numpy as np
import umap.umap_ as umap
from sklearn.cluster import KMeans
import supervision as sv
from ultralytics import YOLO

PLAYER_ID = 1
STRIDE = 30
SOURCE_VIDEO_PATH = "..\\fotball_map_pass\\videos\\video2.mp4"
PLAYER_DETECTION_MODEL = YOLO("..\\fotball_map_pass\\models\\weights.onnx")

def extract_crops(source_video_path: str):
  frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)


  crops = []


  for frame in tqdm(frame_generator, desc="Colleccion de cortes de cajas delimitadoras"):
    result = PLAYER_DETECTION_MODEL.predict(frame, imgsz=1792)[0]

    # Extraer las detecciones manualmente
    boxes = result.boxes.xyxy.cpu().numpy()  # Coordenadas de las cajas
    scores = result.boxes.conf.cpu().numpy()  # Puntajes de confianza
    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # IDs de clases
    # Convertir las detecciones a un formato compatible con Supervision
    detections = sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=class_ids
    )

    detections = detections.with_nms(threshold = 0.5, class_agnostic=True)
    detections = detections[detections.class_id == PLAYER_ID]

    players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
    crops += players_crops
  return crops

crops = extract_crops(SOURCE_VIDEO_PATH)

sv.plot_images_grid(crops[:100], grid_size = (10,10)) 



SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(DEVICE)
EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)

BATCH_SIZE = 32

crops = [sv.cv2_to_pillow(crop) for crop in crops]
batches=chunked(crops, BATCH_SIZE)
data = []
with torch.no_grad():
  for batch in tqdm(batches, desc="embeddings extraction"):
    inputs = EMBEDDINGS_PROCESSOR(images=batch, return_tensors="pt").to(DEVICE)
    outputs = EMBEDDINGS_MODEL(**inputs)
    embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
    data.append(embeddings)
data = np.concatenate(data)

REDUCER = umap.UMAP(n_components=3)
CLUSTERING_MODEL = KMeans(n_clusters = 2)

projections =  REDUCER.fit_transform(data)

clusters = CLUSTERING_MODEL.fit_predict(projections)

team_0 =[
    crop
    for crop, cluster
    in zip(crops, clusters)
    if cluster == 1
]

sv.plot_images_grid(team_0[:100], grid_size = (10,10)) 