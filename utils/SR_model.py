import cv2
import numpy as np
import os
import logging
from PIL import Image
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if YOLO is installed
try:
    from ultralytics import YOLO
    YOLO_INSTALLED = True
except ImportError:
    YOLO_INSTALLED = False
    logger.warning("YOLO package not installed. Object detection will be limited.")

# OpenCV Super Resolution Wrapper
class OpenCV_SR:
    def __init__(self, model_path="models/EDSR_x4.pb", scale=4):
        self.sr = None
        if os.path.exists(model_path):
            try:
                self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
                self.sr.readModel(model_path)
                self.sr.setModel("edsr", scale)
                logger.info(f"✅ OpenCV SR model loaded from {model_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load SR model: {e}")
        else:
            logger.warning(f"⚠️ Model weights not found at '{model_path}'. Falling back to resize.")

    def upscale(self, pil_img):
        np_img = np.array(pil_img)
        if self.sr:
            try:
                sr_img = self.sr.upsample(np_img)
                return Image.fromarray(sr_img)
            except Exception as e:
                logger.error(f"Error in SR upscaling: {e}")

        # Fallback → OpenCV resize
        h, w = np_img.shape[:2]
        upscaled = cv2.resize(np_img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        return Image.fromarray(upscaled)

# Detection + Super-Resolution Pipeline
class DetectorSRPipeline:
    def __init__(self, det_model="yolov8n.pt", sr_model="models/EDSR_x4.pb", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.sr = OpenCV_SR(sr_model, scale=4)
        self.detector = None

        if YOLO_INSTALLED:
            try:
                self.detector = YOLO(det_model)
                logger.info("✅ YOLO model loaded successfully.")
            except Exception as e:
                logger.error(f"❌ Failed to load YOLO model: {e}")
                self.detector = None

    def enhance_image(self, image_array, conf=0.3):
        """Enhance image using SR on detected objects, or upscale full image if YOLO not available"""
        enhanced_image = image_array.copy()

        if self.detector is not None:
            try:
                results = self.detector(image_array, conf=conf, verbose=False)
                for r in results:
                    if r.boxes is not None:
                        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                        for x1, y1, x2, y2 in boxes:
                            crop = image_array[y1:y2, x1:x2]
                            if crop.size > 0:
                                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                sr_crop = self.sr.upscale(pil_crop)
                                sr_crop = cv2.cvtColor(np.array(sr_crop), cv2.COLOR_RGB2BGR)
                                sr_resized = cv2.resize(sr_crop, (x2 - x1, y2 - y1))
                                enhanced_image[y1:y2, x1:x2] = sr_resized
            except Exception as e:
                logger.error(f"Error during YOLO detection: {e}")

        else:
            # No YOLO → upscale full image
            pil_img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
            sr_img = self.sr.upscale(pil_img)
            enhanced_image = cv2.cvtColor(np.array(sr_img), cv2.COLOR_RGB2BGR)

        return enhanced_image