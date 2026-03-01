# models/reid_model.py

import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np


class ReIDModel:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_embedding(self, image):
        if image is None or image.size == 0:
            return None

        try:
            img = self.transform(image).unsqueeze(0)

            with torch.no_grad():
                features = self.model(img)

            embedding = features.view(-1).numpy().astype(np.float32)

            # Normalize so cosine similarity works correctly
            # Without this, body cross-similarities are inflated
            norm = np.linalg.norm(embedding)
            if norm == 0:
                return None

            return embedding / norm

        except Exception as e:
            print(f"[ReIDModel] Error: {e}")
            return None