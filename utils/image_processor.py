import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import time

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        # Lazy loading of models to improve startup time
        self.yolo = None
        self.sr_pipeline = None
        self.text_extractor = None
        self.human_class = [0]  # Only person class
    
    def _load_yolo(self):
        if self.yolo is None:
            try:
                from utils.yolo_model import YOLOModel
                self.yolo = YOLOModel()
                logger.info("YOLO model loaded successfully")
            except ImportError as e:
                logger.error(f"Failed to import YOLO model: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                raise
    
    def _load_sr_pipeline(self):
        if self.sr_pipeline is None:
            try:
                from utils.SR_model import DetectorSRPipeline
                self.sr_pipeline = DetectorSRPipeline()
                logger.info("SR pipeline loaded successfully")
            except ImportError as e:
                logger.error(f"Failed to import SR pipeline: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to load SR pipeline: {e}")
                raise
    
    def _load_text_extractor(self):
        if self.text_extractor is None:
            try:
                from utils.text_extractor import TextExtractor
                self.text_extractor = TextExtractor()
                logger.info("Text extractor loaded successfully")
            except ImportError as e:
                logger.error(f"Failed to import text extractor: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to load text extractor: {e}")
                raise

    def process_image(
        self, 
        image_array: np.ndarray, 
        blur_sensitive: bool = True,
        enhance_quality: bool = True,
        extract_objects: bool = True,
        extract_text: bool = True
    ) -> Tuple[np.ndarray, List[Dict], List[Dict], List[Dict]]:
        """
        Optimized image processing pipeline with lazy loading of models.
        
        Args:
            image_array: Input image as numpy array
            blur_sensitive: Whether to blur human faces/regions
            enhance_quality: Whether to enhance image quality
            extract_objects: Whether to extract detected objects
            extract_text: Whether to extract text from image
            
        Returns:
            Tuple of (processed_image, detections, cropped_objects, text_data)
        """
        try:
            if image_array is None or image_array.size == 0:
                raise ValueError("Empty image array")
            
            logger.info(f"Processing image with shape: {image_array.shape}")
            processed_image = image_array.copy()
            detections = []
            cropped_objects = []
            text_data = []
            
            # Load only required models
            if blur_sensitive or extract_objects:
                logger.info("Loading YOLO model for object detection")
                self._load_yolo()
                detections = self.yolo.detect_objects(image_array)
                logger.info(f"Detected {len(detections)} objects")
            
            # Extract objects if requested
            if extract_objects and detections:
                logger.info("Extracting objects from image")
                cropped_objects = self._crop_objects(image_array, detections)
                logger.info(f"Cropped {len(cropped_objects)} objects")
            
            # Extract text if requested
            if extract_text:
                logger.info("Loading text extractor for OCR")
                self._load_text_extractor()
                text_data = self.text_extractor.extract_text(image_array)
                logger.info(f"Extracted {len(text_data)} text elements")
            
            # Blur sensitive areas if requested
            if blur_sensitive and detections:
                logger.info("Blurring sensitive regions")
                processed_image = self._blur_human_regions(processed_image, detections)
            
            # Enhance image quality if requested
            if enhance_quality:
                logger.info("Loading SR pipeline for image enhancement")
                self._load_sr_pipeline()
                enhanced_image = self.sr_pipeline.enhance_image(processed_image)
                
                # Additional quality enhancement
                enhanced_image = self.enhance_image_quality(enhanced_image)
                processed_image = enhanced_image
                logger.info("Image enhancement completed")
            
            return processed_image, detections, cropped_objects, text_data

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            # Return original image and empty results on error
            return image_array, [], [], []

    def _crop_objects(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Crop detected objects from the image.
        
        Args:
            image: Original image as numpy array
            detections: List of detection dictionaries with bbox information
            
        Returns:
            List of cropped objects with metadata
        """
        cropped_objects = []
        h, w = image.shape[:2]

        for i, det in enumerate(detections):
            bbox = det.get('bbox')
            if bbox is None or len(bbox) != 4:
                continue

            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1:
                    cropped_img = image[y1:y2, x1:x2].copy()
                    cropped_objects.append({
                        'image_bytes': cropped_img,
                        'bbox': [x1, y1, x2, y2],
                        'class': det.get('class'),
                        'label': det.get('label', f'object_{i}'),
                        'confidence': det.get('confidence', 0.0)
                    })
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing bbox {bbox}: {e}")
                continue

        return cropped_objects

    def _blur_human_regions(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Blur human regions in the image for privacy protection using mosaic effect.
        
        Args:
            image: Input image as numpy array
            detections: List of detection dictionaries
            
        Returns:
            Image with blurred human regions
        """
        h, w = image.shape[:2]
        blurred = image.copy()

        for det in detections:
            # Only process human detections
            if det.get('class') not in self.human_class:
                continue
            if det.get('confidence', 0) <= 0.3:
                continue

            bbox = det.get('bbox')
            if bbox is None or len(bbox) != 4:
                continue

            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                roi = blurred[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # Apply mosaic effect for better privacy protection
                # 1. First, resize to very small size
                mosaic_level = 15  # Adjust for more/less blurring
                h_roi, w_roi = roi.shape[:2]
                roi_small = cv2.resize(roi, (mosaic_level, mosaic_level), 
                                      interpolation=cv2.INTER_LINEAR)
                
                # 2. Then resize back to original size
                roi_mosaic = cv2.resize(roi_small, (w_roi, h_roi), 
                                       interpolation=cv2.INTER_NEAREST)
                
                # 3. Apply the mosaic to the original image
                blurred[y1:y2, x1:x2] = roi_mosaic
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing bbox for blurring {bbox}: {e}")
                continue

        return blurred

    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality using various techniques.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Enhanced image
        """
        if image is None or image.size == 0:
            return image

        # Ensure we're working with uint8 before LAB conversion
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        enhanced = image.copy()

        # LAB color contrast enhancement
        if len(enhanced.shape) == 3 and enhanced.shape[2] == 3:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            lab = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Sharpen carefully with reduced kernel strength
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        # Denoise after sharpening
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Final contrast/brightness tweak
        alpha = 1.1  # Contrast control (1.0-3.0)
        beta = 10    # Brightness control (0-100)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
        
        return enhanced

    def process_image_step_by_step(self, image_array: np.ndarray, record_id: str = None):
        """
        Process image step by step for background processing with progress tracking.
        
        Args:
            image_array: Input image as numpy array
            record_id: Optional record ID for progress tracking
            
        Returns:
            Dictionary with processing results
        """
        results = {
            'processed_image': None,
            'detections': [],
            'cropped_objects': [],
            'text_data': []
        }
        
        try:
            # Step 1: Object Detection
            self._load_yolo()
            results['detections'] = self.yolo.detect_objects(image_array)
            
            # Step 2: Blur sensitive regions
            blurred_image = self._blur_human_regions(image_array, results['detections'])
            
            # Step 3: Image Enhancement
            self._load_sr_pipeline()
            enhanced_image = self.sr_pipeline.enhance_image(blurred_image)
            results['processed_image'] = self.enhance_image_quality(enhanced_image)
            
            # Step 4: Text Extraction
            self._load_text_extractor()
            results['text_data'] = self.text_extractor.extract_text(image_array)
            
            # Step 5: Crop Objects
            results['cropped_objects'] = self._crop_objects(image_array, results['detections'])
            
        except Exception as e:
            logger.error(f"Error in step-by-step processing: {e}", exc_info=True)
            
        return results

# Utility function for testing
def test_processor():
    """Test function for the ImageProcessor class"""
    processor = ImageProcessor()
    
    # Create a test image
    test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    # Process the image
    start_time = time.time()
    processed, detections, cropped, text = processor.process_image(
        test_image, 
        blur_sensitive=True,
        enhance_quality=True,
        extract_objects=True,
        extract_text=True
    )
    end_time = time.time()
    
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Detections: {len(detections)}")
    print(f"Cropped objects: {len(cropped)}")
    print(f"Text elements: {len(text)}")
    
    return processed, detections, cropped, text

if __name__ == "__main__":
    # Run test if executed directly
    test_processor()