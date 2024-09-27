import numpy as np
import cv2 as cv
from ultralytics import YOLO

# Cargar el modelo YOLO entrenado
model = YOLO('..\\fotball_map_pass\\models\\weights.onnx')

drawing = False  # true if mouse is pressed
src_x, src_y = -1, -1
dst_x, dst_y = -1, -1

src_list = []
dst_list = []

# Configurar el tamaño de entrada (asumiendo 1792x1792 según tu configuración anterior)
target_size = (1792, 1792)

# mouse callback function para src (imagen capturada del video)
def select_points_src(event, x, y, flags, param):
    global src_x, src_y, drawing
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        src_x, src_y = x, y
        cv.circle(src_copy, (x, y), 5, (0, 0, 255), -1)  # Color de los puntos de referencia
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False

# mouse callback function para dst (cancha virtualizada)
def select_points_dst(event, x, y, flags, param):
    global dst_x, dst_y, drawing
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        dst_x, dst_y = x, y
        cv.circle(dst_copy, (x, y), 5, (0, 0, 255), -1)  # Color de los puntos de referencia
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False

def normalize_points(points, shape):
    """Normaliza los puntos a las dimensiones de la imagen."""
    height, width = shape[:2]
    return np.array([[x / width, y / height] for (x, y) in points])

def get_plan_view(src, dst):
    # Asegurarse de que las dimensiones sean correctas
    print(f"Dimensiones de src: {src.shape}, Dimensiones de dst: {dst.shape}")
    
    # Calcular la homografía usando los puntos seleccionados
    src_pts = np.array(src_list).reshape(-1,1,2)
    dst_pts = np.array(dst_list).reshape(-1,1,2)
    
    # Imprimir puntos para ver si están bien seleccionados
    print(f"Puntos de src: {src_pts}, Puntos de dst: {dst_pts}")
    
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    print("Matriz de homografía H:")
    print(H)
    
    plan_view = cv.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))
    return plan_view, H  # Retornar ambos valores: la imagen proyectada (plan_view) y la homografía (H)

def transform_detections(detections, H, src_shape):
    """Transforma las coordenadas de las detecciones utilizando la homografía."""
    transformed_detections = []
    height, width = src_shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = detection[:4]  # Coordenadas de la detección
        class_id = detection[4]  # Identificar si es jugador o balón
        
        # Obtener el centro de la detección
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Ajustar las coordenadas para que correspondan al tamaño de la imagen original
        point = np.array([[center_x, center_y]], dtype='float32').reshape(-1, 1, 2)
        
        # Aplicar la homografía
        transformed_point = cv.perspectiveTransform(point, H)

        # Imprimir las coordenadas transformadas
        print(f"Detección original: {center_x}, {center_y} -> Proyectado: {transformed_point}")
        
        transformed_detections.append((transformed_point[0][0], class_id))  # Guardar coordenadas transformadas y la clase
    return transformed_detections

def draw_detections_on_field(detections, field_img):
    """Dibuja las detecciones en la cancha virtualizada."""
    for point, class_id in detections:
        x, y = point
        # Dibujar la pelota en rojo y los jugadores en verde
        if class_id == 0:  # Asumiendo que la clase 0 es la pelota
            cv.circle(field_img, (int(x), int(y)), 5, (0, 0, 255), -1)  # Rojo para la pelota
        else:
            cv.circle(field_img, (int(x), int(y)), 5, (0, 255, 0), -1)  # Verde para jugadores
    return field_img

# Cargar la cancha virtualizada
dst = cv.imread('..\\fotball_map_pass\\videos\\dst.jpg', -1)
if dst is None:
    print("Error al cargar la imagen de la cancha virtualizada.")
    exit()
dst_copy = dst.copy()

# Crear las ventanas y setear los callbacks de mouse para ambas imágenes
cv.namedWindow('dst')
cv.setMouseCallback('dst', select_points_dst)

# Cargar el video
cap = cv.VideoCapture('..\\fotball_map_pass\\videos\\pov1.mp4')
ret, src = cap.read()
if not ret:
    print("Error al cargar el video.")
    exit()

# Capturar un frame para selección de puntos
src_copy = src.copy()
cv.namedWindow('src')
cv.setMouseCallback('src', select_points_src)

# Ciclo para seleccionar los puntos y calcular la homografía
while True:
    cv.imshow('src', src_copy)
    cv.imshow('dst', dst_copy)
    k = cv.waitKey(1) & 0xFF
    if k == ord('s'):
        print('save points')
        src_list.append([src_x, src_y])
        dst_list.append([dst_x, dst_y])
        print("src points:")
        print(src_list)
        print("dst points:")
        print(dst_list)
    elif k == ord('h'):
        print('create plan view')
        plan_view, H = get_plan_view(src, dst)  # Obtener la homografía
        if plan_view is not None:
            cv.imshow("plan view", plan_view)
        else:
            print("No se pudo mostrar la vista planificada.")
    elif k == ord('d'):
        # Procesar el video cuadro por cuadro y aplicar detecciones
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Realizar detecciones
            results = model.predict(source=frame, imgsz=target_size)
            
            # Obtener las coordenadas de detección (jugadores y balón)
            detections = []
            for result in results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()  # Coordenadas [x1, y1, x2, y2]
                    class_id = box.cls[0].cpu().numpy()  # Identificar la clase (jugador o balón)
                    detections.append(np.append(xyxy, class_id))  # Agregar clase al final

            # Transformar las detecciones a la cancha virtualizada
            transformed_detections = transform_detections(detections, H, frame.shape)

            # Dibujar las detecciones en la cancha virtualizada
            field_with_detections = draw_detections_on_field(transformed_detections, dst.copy())
            cv.imshow('Detections on Field', field_with_detections)

            # Mostrar el fotograma con las detecciones
            frame_with_detections = frame.copy()
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cv.rectangle(frame_with_detections, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv.imshow('Video with Detections', frame_with_detections)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    elif k == 27:  # Tecla 'Esc' para salir
        break

cv.destroyAllWindows()
