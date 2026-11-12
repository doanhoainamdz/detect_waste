"""YOLO detection with explicit confidence and device settings."""

from ultralytics import YOLO

from .config import YOLO_WEIGHTS, check_weights


class YOLODetector:
    def __init__(self, model_path=YOLO_WEIGHTS, conf_threshold=0.25, device="cpu"):
        if not 0 <= conf_threshold <= 1:
            raise ValueError("conf_threshold must be between 0 and 1")
        self.model = YOLO(str(check_weights(model_path)))
        self.conf_threshold = conf_threshold
        self.device = device
        self.class_names = self.model.names

    def detect(self, frame):
        result = self.model.predict(
            source=frame, conf=self.conf_threshold, device=self.device, verbose=False,
        )[0]
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append({
                "bbox": box.xyxy[0].tolist(),
                "confidence": float(box.conf[0]),
                "class_id": class_id,
                "label": self.class_names[class_id],
            })
        return detections
