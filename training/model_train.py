from ultralytics import YOLO
import os
import torch

current_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_path, "models/pretrained_models/yolov8n-seg.pt")
data_path = os.path.join(current_path, "data/data.yaml")
save_project= os.path.join(current_path, "runs/yolov8")
device = None

#Libreria para mac, cpu o tarjeta NVIDIA
if torch.backends.mps.is_available():
    device = torch.device("mps")

if torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")

#con CPU
model = YOLO(model_path).to(device)

# Leer el modelo YOLOv8

def main():
    # Entrenar el modelo
    model.train(data=data_path, epochs=300, batch=64, imgsz=640, patinence=10, task='segment', device=0,
                project=save_project, verbose=True, plots=True)

if __name__ == '__main__':
    main()