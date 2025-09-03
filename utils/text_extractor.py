import cv2
import easyocr
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TextExtractor:
    def __init__(self, languages=['en']):
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(languages)
    
    def extract_text(self, image_array):
        """Extract text from image array"""
        try:
            results = self.reader.readtext(image_array)
            text_data = []

            for result in results:
                # Handle different return formats safely
                if len(result) == 3:
                    bbox, text, confidence = result
                elif len(result) == 2:
                    bbox, text = result
                    confidence = 0.5  # default confidence if missing
                else:
                    continue  # skip unexpected format

                if confidence > 0.3:
                    # Ensure bbox points are integers
                    bbox_list = [[int(p[0]), int(p[1])] for p in bbox]
                    text_data.append({
                        'text': text,
                        'confidence': round(confidence, 2),
                        'bbox': bbox_list
                    })

            return text_data

        except Exception as e:
            logger.error(f"Error in text extraction: {e}")
            return []