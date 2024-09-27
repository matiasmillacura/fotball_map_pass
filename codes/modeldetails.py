import onnx

# Cargar el modelo ONNX
model = onnx.load('..\\fotball_map_pass\\models\\weights.onnx')
import onnx
import numpy as np
# Mostrar los metadatos del modelo ONNX
if model.metadata_props:
    for prop in model.metadata_props:
        print(f"Propiedad: {prop.key}, Valor: {prop.value}")
else:
    print("No hay metadatos disponibles.")


