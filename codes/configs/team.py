from typing import Generator, Iterable, List, TypeVar
import numpy as np
import supervision as sv
import torch
import umap.umap_ as umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Genera lotes a partir de una secuencia con un tamaño de lote específico.

    Args:
        secuencia (Iterable [V]): la secuencia de entrada que se va a procesar por lotes.
        lote_size (int): el tamaño de cada lote.

    Returns:
        Generador[Lista[V], Ninguno, Ninguno]: un generador que produce lotes de la entrada
            secuencia.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    Un clasificador que utiliza un SiglipVisionModel previamente entrenado para la extracción de características.
    UMAP para reducción de dimensionalidad y KMeans para agrupación.
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extraiga características de una lista de cortes de imágenes utilizando el método previamente entrenado
        Modelo SiglipVision.

        Args:
            cultivos (List[np.ndarray]): Lista de cultivos de imágenes.

        Returns:
            np.ndarray: características extraídas como una matriz numerosa.
        """
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                inputs = self.processor(
                    images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Ajustar el modelo clasificador a una lista de cortes de imágenes.

        Args:
            cultivos (List[np.ndarray]): Lista de cultivos de imágenes.
        """
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predice las etiquetas de los grupos para obtener una lista de cortes de imágenes.

        Argumentos:
            crops (List[np.ndarray]): Lista de cortes de imágenes.

        Devoluciones:
            np.ndarray: etiquetas de clúster previstas.
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)
