"""
Computer Vision Module for KhudaFinder

This module provides computer vision capabilities including:
- Object detection using YOLOv8
- Visualization utilities
"""

# Import available modules
from .detector.yolov8_detector import YOLOv8Detector
from .utils.visualizer import Visualizer

__version__ = "0.1.0"
__all__ = [
    "YOLOv8Detector",
    "Visualizer"
]
