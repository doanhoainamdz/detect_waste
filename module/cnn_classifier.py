"""ConvNeXt Tiny classifier matching the ImageNet-normalized training pipeline."""

import cv2
import torch
from PIL import Image
from torchvision import models, transforms

from .config import CLASS_NAMES, CNN_WEIGHTS, check_weights


class CNNClassifier:
    def __init__(self, model, class_names=CLASS_NAMES, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.class_names = tuple(class_names)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @classmethod
    def from_checkpoint(cls, path=CNN_WEIGHTS, class_names=CLASS_NAMES, device="cpu"):
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, len(class_names))
        state = torch.load(check_weights(path), map_location="cpu", weights_only=True)
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
        model.load_state_dict(state)
        return cls(model, class_names, device)

    def predict_with_confidence(self, image):
        if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("CNN expects a non-empty BGR image with three channels")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)
            confidence, predicted = probabilities.max(dim=1)
        return self.class_names[predicted.item()], float(confidence.item())

    def predict(self, image):
        return self.predict_with_confidence(image)[0]
