import sys
import cv2
import numpy as np
from configs.soccer import SoccerPitchConfiguration
import supervision as sv
from typing import Optional, List


def draw_pitch(
    config: SoccerPitchConfiguration,
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1
) -> np.ndarray:
    """
    Draws a soccer pitch with specified dimensions, colors, and scale, including node numbers.

    Dibuja un campo de fútbol con dimensiones, colores y escala específicos, incluidos los números de nodo.

    Argumentos:
        config (SoccerPitchConfiguration): objeto de configuración que contiene el
            Dimensiones y distribución del terreno de juego.
        background_color (sv.Color, opcional): Color del fondo del tono.
            El valor predeterminado es sv.Color(34, 139, 34).
        line_color (sv.Color, opcional): Color de las líneas de paso.
            El valor predeterminado es sv.Color.WHITE.
        padding (int, opcional): relleno alrededor del tono en píxeles.
            El valor predeterminado es 50.
        line_thickness (int, opcional): Grosor de las líneas de paso en píxeles.
            El valor predeterminado es 4.
        point_radius (int, opcional): Radio de los puntos de penalización en píxeles.
            El valor predeterminado es 8.
        escala (flotante, opcional): factor de escala para las dimensiones de paso.
            El valor predeterminado es 0,1.

    Devoluciones:
        np.ndarray: Imagen del campo de fútbol.
    """
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    pitch_image = np.ones(
        (scaled_width + 2 * padding,
         scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    for start, end in config.edges:
        point1 = (int(config.vertices[start - 1][0] * scale) + padding,
                  int(config.vertices[start - 1][1] * scale) + padding)
        point2 = (int(config.vertices[end - 1][0] * scale) + padding,
                  int(config.vertices[end - 1][1] * scale) + padding)
        cv2.line(
            img=pitch_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )

    centre_circle_center = (
        scaled_length // 2 + padding,
        scaled_width // 2 + padding
    )
    cv2.circle(
        img=pitch_image,
        center=centre_circle_center,
        radius=scaled_circle_radius,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )

    penalty_spots = [
        (
            scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        ),
        (
            scaled_length - scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        )
    ]
    for spot in penalty_spots:
        cv2.circle(
            img=pitch_image,
            center=spot,
            radius=point_radius,
            color=line_color.as_bgr(),
            thickness=-1
        )

    for index, vertex in enumerate(config.vertices):
        x, y = int(vertex[0] * scale) + padding, int(vertex[1] * scale) + padding
        cv2.putText(
            img=pitch_image,
            text=str(index + 1),
            org=(x, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,  # Tamaño del texto
            color=(0, 0, 0),  # Color del texto (Negro)
            thickness=1,
            lineType=cv2.LINE_AA
        )

        return pitch_image

def draw_points_on_pitch(
    config: SoccerPitchConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Dibuja puntos en el campo de fútbol.

    Argumentos:
        config (SoccerPitchConfiguration): objeto de configuración que contiene el
            Dimensiones y distribución del terreno de juego.
        xy (np.ndarray): conjunto de puntos a dibujar, con cada punto representado por
            sus coordenadas (x, y).
        face_color (sv.Color, opcional): Color de las caras de los puntos.
            El valor predeterminado es sv.Color.RED.
        edge_color (sv.Color, opcional): Color de los bordes de los puntos.
            El valor predeterminado es sv.Color.BLACK.
        radio (int, opcional): Radio de los puntos en píxeles.
            El valor predeterminado es 10.
        espesor (int, opcional): espesor de los bordes del punto en píxeles.
            El valor predeterminado es 2.
        padding (int, opcional): relleno alrededor del tono en píxeles.
            El valor predeterminado es 50.
        escala (flotante, opcional): factor de escala para las dimensiones de paso.
            El valor predeterminado es 0,1.
        paso (Opcional[np.ndarray], opcional): imagen de paso existente para dibujar puntos.
            Si no hay ninguno, se creará una nueva propuesta. El valor predeterminado es Ninguno.

    Devoluciones:
        np.ndarray: Imagen del campo de fútbol con puntos dibujados.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    # Verificar si las coordenadas transformadas son válidas
    if xy is None or len(xy) == 0:
        print("Advertencia: No se encontraron coordenadas para proyectar.")
        return pitch

    for point in xy:
        # Verificar que las coordenadas no sean NaN o valores inválidos
        if np.isnan(point).any():
            print("Advertencia: Coordenadas no válidas detectadas:", point)
            continue

        # Aplicar escalado y ajuste para posicionar los puntos
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        
        # Dibujar el punto en la cancha
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness
        )

    return pitch



def draw_paths_on_pitch(
    config: SoccerPitchConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Dibuja caminos en un campo de fútbol.

    Argumentos:
        config (SoccerPitchConfiguration): objeto de configuración que contiene el
            Dimensiones y distribución del terreno de juego.
        rutas (List[np.ndarray]): Lista de rutas, donde cada ruta es una matriz de (x, y)
            coordenadas.
        color (sv.Color, opcional): Color de los trazados.
            El valor predeterminado es sv.Color.WHITE.
        espesor (int, opcional): Grosor de los trazados en píxeles.
            El valor predeterminado es 2.
        padding (int, opcional): relleno alrededor del tono en píxeles.
            El valor predeterminado es 50.
        escala (flotante, opcional): factor de escala para las dimensiones de paso.
            El valor predeterminado es 0,1.
        paso (Opcional[np.ndarray], opcional): imagen de paso existente para dibujar trazados.
            Si no hay ninguno, se creará una nueva propuesta. El valor predeterminado es Ninguno.

    Devoluciones:
        np.ndarray: Imagen del campo de fútbol con caminos dibujados.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    for path in paths:
        scaled_path = [
            (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding
            )
            for point in path if point.size > 0
        ]

        if len(scaled_path) < 2:
            continue

        for i in range(len(scaled_path) - 1):
            cv2.line(
                img=pitch,
                pt1=scaled_path[i],
                pt2=scaled_path[i + 1],
                color=color.as_bgr(),
                thickness=thickness
            )

        return pitch


def draw_pitch_voronoi_diagram(
    config: SoccerPitchConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.RED,
    team_2_color: sv.Color = sv.Color.WHITE,
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Dibuja un diagrama de Voronoi en un campo de fútbol que representa las áreas de control de dos
    equipos.

    Argumentos:
        config (SoccerPitchConfiguration): objeto de configuración que contiene el
            Dimensiones y distribución del terreno de juego.
        team_1_xy (np.ndarray): Matriz de coordenadas (x, y) que representan las posiciones
            de jugadores del equipo 1.
        team_2_xy (np.ndarray): Matriz de coordenadas (x, y) que representan las posiciones
            de jugadores del equipo 2.
        team_1_color (sv.Color, opcional): Color que representa el área de control de
            equipo 1. El valor predeterminado es sv.Color.RED.
        team_2_color (sv.Color, opcional): Color que representa el área de control de
            equipo 2. El valor predeterminado es sv.Color.WHITE.
        opacidad (flotante, opcional): opacidad de la superposición del diagrama de Voronoi.
            El valor predeterminado es 0,5.
        padding (int, opcional): relleno alrededor del tono en píxeles.
            El valor predeterminado es 50.
        escala (flotante, opcional): factor de escala para las dimensiones de paso.
            El valor predeterminado es 0,1.
        tono (Opcional[np.ndarray], opcional): Imagen de tono existente para dibujar el
            Diagrama de Voronoi encendido. Si no hay ninguno, se creará una nueva propuesta. El valor predeterminado es Ninguno.

    Devoluciones:
        np.ndarray: Imagen del campo de fútbol con el diagrama de Voronoi superpuesto.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    voronoi = np.zeros_like(pitch, dtype=np.uint8)

    team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    y_coordinates, x_coordinates = np.indices((
        scaled_width + 2 * padding,
        scaled_length + 2 * padding
    ))

    y_coordinates -= padding
    x_coordinates -= padding

    def calculate_distances(xy, x_coordinates, y_coordinates):
        return np.sqrt((xy[:, 0][:, None, None] * scale - x_coordinates) ** 2 +
                       (xy[:, 1][:, None, None] * scale - y_coordinates) ** 2)

    distances_team_1 = calculate_distances(team_1_xy, x_coordinates, y_coordinates)
    distances_team_2 = calculate_distances(team_2_xy, x_coordinates, y_coordinates)

    min_distances_team_1 = np.min(distances_team_1, axis=0)
    min_distances_team_2 = np.min(distances_team_2, axis=0)

    control_mask = min_distances_team_1 < min_distances_team_2

    voronoi[control_mask] = team_1_color_bgr
    voronoi[~control_mask] = team_2_color_bgr

    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

    return overlay