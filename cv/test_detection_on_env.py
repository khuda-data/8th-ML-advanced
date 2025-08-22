#!/usr/bin/env python3
"""
CV 모듈 테스트 파일
"""

import sys
import os

print("Python 경로:")
for path in sys.path:
    print(f"  {path}")

print(f"\n현재 작업 디렉토리: {os.getcwd()}")

try:
    print("\nOpenCV 테스트:")
    import cv2
    print(f"OpenCV 버전: {cv2.__version__}")
except ImportError as e:
    print(f"OpenCV import 실패: {e}")

try:
    print("\nNumPy 테스트:")
    import numpy as np
    print(f"NumPy 버전: {np.__version__}")
except ImportError as e:
    print(f"NumPy import 실패: {e}")

try:
    print("\nCV 모듈 테스트:")
    from cv import YOLOv8Detector, Visualizer
    print("CV 모듈 import 성공!")
    print(f"사용 가능한 클래스: YOLOv8Detector, Visualizer")
except ImportError as e:
    print(f"CV 모듈 import 실패: {e}")
    print("프로젝트 루트 디렉토리에서 실행하고 있는지 확인하세요.")
