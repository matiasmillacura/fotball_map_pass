import cv2
from ultralytics import YOLO


# Cargar el modelo entrenado
model = YOLO('..\\fotball_map_pass\\models\\weights.onnx')

# Abrir el archivo de video
cap = cv2.VideoCapture('..\\fotball_map_pass\\videos\\corte pov 2.mp4')

# Configurar el tamaño de entrada (asumiendo 1792x1792 según tu configuración anterior)
target_size = (1792, 1792)

# Abrir el archivo de texto para guardar las detecciones
with open('detecciones.txt', 'w') as f:
    
    frame_count = 0  # Contador de fotogramas

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar el fotograma al tamaño esperado por el modelo
        resized_frame = cv2.resize(frame, target_size)

        # Asegurarse de que el fotograma tenga 3 canales (RGB) si es necesario
        if resized_frame.shape[2] != 3:
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)

        # Hacer la detección con YOLO y ByteTrack (en este caso usando BoTSORT)
        results = model.track(source=resized_frame,iou=0.5,conf=0.4, imgsz=target_size, tracker="A:\\ultralytics\\ultralytics\\cfg\\trackers\\botsort.yaml", persist=True)

        # Escribir los resultados de detección en el archivo .txt
        f.write(f"Frame {frame_count}:\n")
        for result in results:
            for box in result.boxes:
                # Obtener las coordenadas de la caja y la confianza
                xyxy = box.xyxy[0].cpu().numpy()  # Coordenadas [x1, y1, x2, y2]
                score = box.conf[0].cpu().numpy()  # Confianza
                class_id = box.cls[0].cpu().numpy()  # Clase (jugador o balón)
                
                # Asignar nombres de clase
                class_name = "JUGADOR" if class_id == 1 else "BALON"

                # Formatear las coordenadas y confianza
                f.write(f"  Clase: {class_name}, Confianza: {score:.2f}, Coordenadas: ({xyxy[0]:.2f}, {xyxy[1]:.2f}, {xyxy[2]:.2f}, {xyxy[3]:.2f})\n")

        # Dibujar las anotaciones con texto pequeño sin usar `result.plot()`
        font_scale = 0.5  # Escala de la fuente (más pequeño)
        thickness = 1  # Grosor del texto
        color = (255, 0, 0)  # Color del texto (azul)

        # Dibujar las cajas delimitadoras y la ID manualmente
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()  # Coordenadas [x1, y1, x2, y2]
                track_id = box.id  # ID del seguimiento (si está disponible)
                x1, y1, x2, y2 = map(int, xyxy)  # Convertir a enteros

                # Dibujar caja delimitadora
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Dibujar la ID encima de la caja con texto más pequeño
                if track_id is not None:
                    cv2.putText(resized_frame, f"ID: {int(track_id)}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        # Mostrar el fotograma procesado en tiempo real
        cv2.imshow("Real-Time Detection", resized_frame)

        # Romper el bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Incrementar el contador de fotogramas
        frame_count += 1

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
