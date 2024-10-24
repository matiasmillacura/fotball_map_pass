from ultralytics import YOLO

model = YOLO('..\\fotball_map_pass\\models\\weights.onnx')

model.val(data = "C:\\Users\\Matias\\Documents\\GitHub\\fotball_map_pass\\data\\dataset2\\data.yaml", iou=0.5, imgsz = 1792, name="inferencia")