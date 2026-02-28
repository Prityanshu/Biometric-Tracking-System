import cv2
import torch
from ultralytics import YOLO

print("OpenCV:", cv2.__version__)
print("Torch CUDA:", torch.cuda.is_available())

model = YOLO("yolov8n.pt")
print("YOLO ready")