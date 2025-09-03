import cv2
import numpy as np
import os
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class YOLOModel:
    def __init__(self, model_name='yolov8n.pt'):
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load YOLO model with error handling"""
        try:
            self.model = YOLO(self.model_name)
            logger.info("YOLO model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            return False
    
    def detect_objects(self, image):
        """Detect objects in image using YOLO"""
        if self.model is None:
            if not self.load_model():
                return []
        
        try:
            # Run YOLO inference
            results = self.model(image)
            detections = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = self.model.names[cls]
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': cls,
                            'label': label
                        })
            
            return detections
        except Exception as e:
            logger.error(f"Error during YOLO inference: {e}")
            return []