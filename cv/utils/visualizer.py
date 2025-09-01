import cv2
import numpy as np
from typing import List, Dict, Any

def draw_tracked_objects(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    추적된 객체들의 정보(ID 포함)를 이미지에 그림
    """
    for det in detections:
        box = det['box']
        label = f"ID {det.get('id', '?')}: {det['label']} ({det['confidence']:.2f})"
        x1, y1, x2, y2 = box
        
        # BGR 색상 정의
        color = (0, 255, 0) # 초록색
        if 'agent' in det['label']:
            color = (255, 0, 0) # 파란색
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image