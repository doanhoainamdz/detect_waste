"""Behavior checks independent of the large model checkpoints."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from module.cnn_classifier import CNNClassifier
from module.config import check_weights
from module.hybrid_model import HybridModel
from module.yolo_detector import YOLODetector


def detection(confidence=0.9, bbox=(-4, -2, 12, 14)):
    return {"bbox": bbox, "confidence": confidence, "class_id": 0, "label": "battery"}


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((10, 10, 3), dtype=np.uint8)
        self.yolo = Mock()
        self.cnn = Mock()
        self.cnn.predict_with_confidence.return_value = ("pcb", 0.8)
        self.model = HybridModel(self.yolo, self.cnn, conf_threshold=0.6)

    def test_high_confidence_keeps_yolo_label_and_clamps_box(self):
        self.yolo.detect.return_value = [detection()]
        result = self.model.process(self.frame)[0]
        self.assertEqual(result["bbox"], (0, 0, 10, 10))
        self.assertEqual(result["label"], "battery")
        self.assertEqual(result["source"], "YOLO")
        self.cnn.predict_with_confidence.assert_not_called()

    def test_low_confidence_uses_valid_crop_and_preserves_both_scores(self):
        self.yolo.detect.return_value = [detection(0.4, (1, 2, 8, 9))]
        result = self.model.process(self.frame)[0]
        self.assertEqual(result["label"], "pcb")
        self.assertEqual(result["source"], "CNN")
        self.assertEqual(result["confidence"], 0.8)
        self.assertEqual(result["detection_confidence"], 0.4)
        self.assertEqual(result["classification_confidence"], 0.8)
        crop = self.cnn.predict_with_confidence.call_args.args[0]
        self.assertEqual(crop.shape, (7, 7, 3))

    def test_threshold_equality_keeps_yolo(self):
        self.yolo.detect.return_value = [detection(0.6)]
        self.assertEqual(self.model.process(self.frame)[0]["source"], "YOLO")

    def test_yolo_only_does_not_require_cnn(self):
        self.yolo.detect.return_value = [detection(0.3)]
        self.assertEqual(HybridModel(self.yolo).process(self.frame)[0]["source"], "YOLO")

    def test_outside_or_reversed_boxes_are_skipped(self):
        self.yolo.detect.return_value = [detection(0.3, (20, 20, 30, 30)), detection(0.3, (8, 8, 2, 2))]
        self.assertEqual(self.model.process(self.frame), [])
        self.cnn.predict_with_confidence.assert_not_called()

    def test_no_detection_does_not_classify_background(self):
        self.yolo.detect.return_value = []
        self.assertEqual(self.model.process(self.frame), [])
        self.cnn.predict_with_confidence.assert_not_called()

    def test_invalid_frame_and_threshold_are_rejected(self):
        for frame in (None, np.empty((0, 0, 3)), np.zeros((10, 10))):
            with self.assertRaises(ValueError):
                self.model.process(frame)
        with self.assertRaises(ValueError):
            HybridModel(self.yolo, self.cnn, 1.1)


class CaptureClassifier(torch.nn.Module):
    def forward(self, inputs):
        self.last_input = inputs
        return torch.tensor([[0.0, 2.0]], device=inputs.device)


class ClassifierTests(unittest.TestCase):
    def test_rgb_normalization_cpu_and_class_confidence(self):
        model = CaptureClassifier()
        classifier = CNNClassifier(model, ("battery", "pcb"), "cpu")
        # OpenCV's pure blue pixel must become RGB [0, 0, 255].
        blue_bgr = np.full((8, 8, 3), [255, 0, 0], dtype=np.uint8)
        label, confidence = classifier.predict_with_confidence(blue_bgr)
        self.assertEqual(label, "pcb")
        self.assertAlmostEqual(confidence, 0.880797, places=5)
        self.assertFalse(model.training)
        self.assertEqual(tuple(model.last_input.shape), (1, 3, 224, 224))
        expected = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, (1 - 0.406) / 0.225])
        torch.testing.assert_close(model.last_input[0, :, 0, 0], expected)
        self.assertEqual(model.last_input.device.type, "cpu")

    def test_empty_crop_is_rejected(self):
        classifier = CNNClassifier(CaptureClassifier(), ("a", "b"), "cpu")
        with self.assertRaises(ValueError):
            classifier.predict(np.empty((0, 0, 3), dtype=np.uint8))


class DetectorTests(unittest.TestCase):
    def test_threshold_device_and_actual_class_label_are_forwarded(self):
        detector = YOLODetector.__new__(YOLODetector)
        detector.conf_threshold = 0.15
        detector.device = "cpu"
        detector.class_names = {2: "keyboard"}
        detector.model = Mock()
        box = SimpleNamespace(conf=[0.7], cls=[2], xyxy=torch.tensor([[1, 2, 5, 6]]))
        detector.model.predict.return_value = [Mock(boxes=[box])]
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        result = detector.detect(frame)[0]
        self.assertEqual(result["label"], "keyboard")
        self.assertEqual(result["bbox"], [1, 2, 5, 6])
        kwargs = detector.model.predict.call_args.kwargs
        self.assertIs(kwargs["source"], frame)
        self.assertEqual(kwargs["conf"], 0.15)
        self.assertEqual(kwargs["device"], "cpu")


class WeightsTests(unittest.TestCase):
    def test_missing_weights_and_lfs_pointer_have_actionable_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            with self.assertRaisesRegex(FileNotFoundError, "git lfs pull"):
                check_weights(path)
            path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:test\nsize 123\n")
            with self.assertRaisesRegex(ValueError, "git lfs pull"):
                check_weights(path)


if __name__ == "__main__":
    unittest.main()
