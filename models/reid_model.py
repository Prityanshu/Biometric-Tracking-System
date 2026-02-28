# models/reid_model.py

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

class ReIDModel:
    def __init__(self):
        # Load pretrained ResNet50
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # remove last layer
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_embedding(self, image):
        img = self.transform(image).unsqueeze(0)  # add batch dim

        with torch.no_grad():
            features = self.model(img)

        return features.view(-1).numpy()