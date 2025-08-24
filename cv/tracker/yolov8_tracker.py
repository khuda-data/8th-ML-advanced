# cv/tracker/yolov8_tracker.py

from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any

class YOLOv8Tracker:
    """
    YOLOv8 모델을 사용하여 이미지에서 객체를 '추적'하는 클래스.
    탐지된 각 객체에 고유한 ID를 부여하고 프레임 간에 유지.
    """
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.last_results = None

    def track(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        주어진 이미지에서 객체를 추적하고, 각 객체에 ID를 부여하여 반환.
        """
        results = self.model.track(image, persist=True, verbose=False)
        self.last_results = results[0]

        detections = []
        if self.last_results.boxes.id is None:
            return []

        for box in self.last_results.boxes:
            tracker_id = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = self.model.names[class_id]

            detections.append({
                'id': tracker_id,
                'box': (x1, y1, x2, y2),
                'label': label,
                'confidence': confidence
            })
        
        return detections