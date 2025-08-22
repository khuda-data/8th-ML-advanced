"""
Visualization Utilities for Computer Vision

This module provides utilities for visualizing detection results and other CV outputs.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

class Visualizer:
    """
    Utility class for visualizing computer vision results.
    """
    
    def __init__(self, line_thickness: int = 2, font_scale: float = 0.5):
        """
        Initialize the visualizer.
        
        Args:
            line_thickness: Thickness of bounding box lines
            font_scale: Scale of text font
        """
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
    def draw_detections(self, 
                       image: np.ndarray, 
                       detections: List[Tuple[int, int, int, int, float, int]],
                       class_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Draw detection bounding boxes on an image.
        
        Args:
            image: Input image as numpy array
            detections: List of detections as (x1, y1, x2, y2, confidence, class_id)
            class_names: Optional list of class names
            
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            
            # Choose color based on class_id
            color = self.colors[class_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, self.line_thickness)
            
            # Prepare label text
            if class_names and class_id < len(class_names):
                label = f"{class_names[class_id]}: {confidence:.2f}"
            else:
                label = f"Class {class_id}: {confidence:.2f}"
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
            )
            cv2.rectangle(result_image, (x1, y1 - text_height - 5), 
                         (x1 + text_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 
                       (255, 255, 255), 1)
                       
        return result_image
    
    def show_image(self, image: np.ndarray, window_name: str = "Image", wait_key: bool = True):
        """
        Display an image in a window.
        
        Args:
            image: Image to display
            window_name: Name of the display window
            wait_key: Whether to wait for key press
        """
        cv2.imshow(window_name, image)
        if wait_key:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    def save_image(self, image: np.ndarray, filepath: str) -> bool:
        """
        Save an image to file.
        
        Args:
            image: Image to save
            filepath: Output file path
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            cv2.imwrite(filepath, image)
            print(f"Image saved to: {filepath}")
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
            
    def __str__(self):
        return f"Visualizer(line_thickness={self.line_thickness}, font_scale={self.font_scale})"
