"""Use CNN classification only for valid YOLO crops below the switch threshold."""


class HybridModel:
    def __init__(self, yolo, cnn=None, conf_threshold=0.6):
        if not 0 <= conf_threshold <= 1:
            raise ValueError("conf_threshold must be between 0 and 1")
        self.yolo = yolo
        self.cnn = cnn
        self.conf_threshold = conf_threshold

    def process(self, frame):
        if frame is None or frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a non-empty BGR image with three channels")
        height, width = frame.shape[:2]
        results = []
        for detection in self.yolo.detect(frame):
            x1, y1, x2, y2 = map(int, detection["bbox"])
            x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
            y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            score = detection["confidence"]
            result = {
                "bbox": (x1, y1, x2, y2),
                "label": detection["label"],
                "confidence": score,
                "detection_confidence": score,
                "source": "YOLO",
            }
            if self.cnn is not None and score < self.conf_threshold:
                label, confidence = self.cnn.predict_with_confidence(frame[y1:y2, x1:x2])
                result.update(label=label, confidence=confidence, source="CNN")
                result["classification_confidence"] = confidence
            results.append(result)
        return results
