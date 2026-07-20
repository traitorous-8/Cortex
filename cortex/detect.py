from dataclasses import dataclass

PPE_CLASSES = [
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "none",
    "Person",
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots"
]


@dataclass
class Detection:
    cls_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        width = max(0.0, self.x2 - self.x1)
        height = max(0.0, self.y2 - self.y1)
        return width * height

    def iou(self, other: 'Detection') -> float:
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)

        inter_area = inter_w * inter_h
        union_area = self.area + other.area - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area


class Detector:
    def __init__(self, weights: str, conf_threshold: float = 0.35):
        self.weights = weights
        self.conf_threshold = conf_threshold
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import torch
            from ultralytics import YOLO
            import functools

            # Monkeypatch torch.load to always use weights_only=False to support older YOLO weights on PyTorch 2.6+
            original_torch_load = torch.load
            @functools.wraps(original_torch_load)
            def load_with_weights_only_false(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)

            torch.load = load_with_weights_only_false
            try:
                self._model = YOLO(self.weights)
            finally:
                torch.load = original_torch_load

        return self._model

    def detect_frame(self, frame):
        """
        Run inference on a single numpy BGR frame.
        """
        results = self.model(frame, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue
                cls_idx = int(box.cls[0])
                # We assume the model is trained with PPE_CLASSES or use YOLO classes?
                # YOLOv8 default model has different classes, but the prompt says
                # PPE_CLASSES list is as follows. We'll assume the model's names attribute maps it
                # or we just use model.names[cls_idx] if it's there.

                cls_name = r.names[cls_idx]
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                detections.append(Detection(
                    cls_name=cls_name,
                    confidence=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2
                ))
        return detections
