from ultralytics import YOLO

# Load your custom trained model
model = YOLO('yolov8n.pt')

# Export to TensorRT .engine format
model.export(format='onnx', imgsz=320, half=True,simplify = True)