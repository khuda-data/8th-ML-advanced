from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any

class YOLOv8Tracker:
    def __init__(self, model_path: str):
        """
        YOLOv8 모델을 로드합니다.

        Args:
            model_path (str): 학습된 YOLOv8 모델 가중치 파일(.pt)의 경로.
        """
        self.model = YOLO(model_path)
        # 추적기는 상태를 유지해야 하므로, 마지막 추적 결과를 저장할 변수 초기화
        self.last_results = None

    def track(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        주어진 이미지에서 객체를 추적하고, 각 객체에 ID를 부여하여 반환합니다.

        Args:
            image (np.ndarray): 추적을 수행할 이미지 (OpenCV 형식, BGR).

        Returns:
            List[Dict[str, Any]]: 추적된 각 객체에 대한 정보 리스트.
            예시: [{'id': 1, 'box': (x1, y1, x2, y2), 'label': 'obstacle', 'confidence': 0.85}, ...]
        """
        # persist=True 옵션은 현재 이미지와 이전 이미지 간의 객체를 연결하기 위해 필수입니다.
        results = self.model.track(image, persist=True, verbose=False)
        self.last_results = results[0] # 첫 번째 결과만 사용

        detections = []
        if self.last_results.boxes.id is None:
            # 추적에 실패했거나, 탐지된 객체가 없는 경우
            return []

        # 추적된 객체들의 정보를 하나씩 추출
        for box in self.last_results.boxes:
            # box.id는 텐서 형태이므로 int로 변환
            tracker_id = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = self.model.names[class_id]

            detections.append({
                'id': tracker_id,          # <--- [중요] 객체 추적 ID 추가!
                'box': (x1, y1, x2, y2),
                'label': label,
                'confidence': confidence
            })
        
        return detections