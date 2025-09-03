import uuid
import cv2
import numpy as np
import base64
import logging
from typing import Optional, Tuple, Any, Dict, List

from flask import (
    Blueprint, render_template, request, redirect,
    url_for, flash, jsonify, current_app
)
from werkzeug.utils import secure_filename
from threading import Thread

from utils.image_processor import ImageProcessor
from models.image_record import ImageRecord

# -----------------------------------------------------------------------------
# Blueprint
# -----------------------------------------------------------------------------
main_bp = Blueprint('main', __name__)

# -----------------------------------------------------------------------------
# Lazy singletons
# -----------------------------------------------------------------------------
_image_processor: Optional[ImageProcessor] = None


def get_image_processor() -> ImageProcessor:
    global _image_processor
    if _image_processor is None:
        _image_processor = ImageProcessor()
        current_app.logger.info("✅ ImageProcessor initialized")
    return _image_processor


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _safe_int(value: Any) -> Optional[int]:
    try:
        # Block explicit "None"/empty strings too
        if value is None:
            return None
        s = str(value).strip()
        if s.lower() == "none" or s == "":
            return None
        return int(s)
    except (ValueError, TypeError):
        return None


def allowed_file(filename: str) -> bool:
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']
    )


def encode_image(img: Any) -> Optional[str]:
    if isinstance(img, np.ndarray):
        success, encoded = cv2.imencode('.jpg', img)
        if success:
            return base64.b64encode(encoded).decode('utf-8')
    elif isinstance(img, (bytes, bytearray)):
        return base64.b64encode(bytes(img)).decode('utf-8')
    return None


def upload_to_supabase(image_bytes: bytes, filename: str, content_type: str = "image/jpeg") -> Optional[str]:
    try:
        bucket = current_app.supabase.storage.from_(current_app.config['SUPABASE_BUCKET_NAME'])
        # Avoid collisions; rely on our own uuid’d names
        _ = bucket.upload(filename, image_bytes, {"content-type": content_type, "upsert": False})
        url = bucket.get_public_url(filename)
        return url or None
    except Exception as e:
        current_app.logger.exception(f"[Supabase] Upload failed for {filename}: {e}")
        return None


# -----------------------------------------------------------------------------
# Background Worker
# -----------------------------------------------------------------------------
def process_in_background(app, record_id: int, image_array_bytes: bytes) -> None:
    """Runs the full pipeline in a thread with proper app context."""
    with app.app_context():
        try:
            # Decode input image
            np_img = np.frombuffer(image_array_bytes, np.uint8)
            image_array = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            if image_array is None:
                raise ValueError("Invalid image in background processing")

            processor = get_image_processor()
            record = ImageRecord.get(record_id)
            if not record:
                raise ValueError(f"Record not found (id={record_id}) in background processing")

            # 1) YOLO
            processor._load_yolo()
            detections = processor.yolo.detect_objects(image_array)
            record.detections = detections or []
            record.detections_count = len(record.detections)
            record.yolo_completed = True
            record.save()

            # 2) Blur sensitive regions
            blurred_image = processor._blur_human_regions(image_array, record.detections)

            # 3) Super-Resolution / Enhancement
            processor._load_sr_pipeline()
            enhanced_image = processor.sr_pipeline.enhance_image(blurred_image)
            enhanced_image = processor.enhance_image_quality(enhanced_image)

            success, encoded_processed = cv2.imencode('.jpg', enhanced_image)
            if not success:
                raise ValueError("Error encoding processed image")
            processed_bytes = encoded_processed.tobytes()
            uid = str(uuid.uuid4())
            processed_filename = f"processed_{uid}.jpg"
            processed_url = upload_to_supabase(processed_bytes, processed_filename)
            if not processed_url:
                raise ValueError("Error uploading processed image")
            record.processed_url = processed_url
            record.sr_completed = True
            record.save()

            # 4) OCR
            processor._load_text_extractor()
            text_data = processor.text_extractor.extract_text(image_array) or []
            record.text_data = text_data
            record.text_elements_count = len(text_data)
            record.ocr_completed = True
            record.save()

            # 5) Crops
            cropped_objects = processor._crop_objects(image_array, record.detections) or []
            cropped_urls: List[Dict[str, Any]] = []
            for i, obj in enumerate(cropped_objects):
                success, enc = cv2.imencode('.jpg', obj.get('image_bytes'))
                if success:
                    cropped_bytes = enc.tobytes()
                    cropped_filename = f"cropped_{record_id}_{uid}_{i}.jpg"
                    cropped_url = upload_to_supabase(cropped_bytes, cropped_filename)
                    if cropped_url:
                        cropped_urls.append({
                            'url': cropped_url,
                            'label': obj.get('label', f'Object {i + 1}'),
                            'confidence': obj.get('confidence', 0.0),
                        })

            record.cropped_objects_urls = cropped_urls
            record.cropped_objects_count = len(cropped_urls)
            record.cropping_completed = True

            # Finalize
            record.db_completed = True
            record.status = 'completed'
            record.save()

        except Exception as e:
            current_app.logger.exception(f"Background processing error for record {record_id}: {e}")
            try:
                record = ImageRecord.get(record_id)
                if record:
                    record.status = 'failed'
                    record.save()
            except Exception:
                current_app.logger.exception("Failed to mark record as failed")


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@main_bp.route('/')
def index():
    return render_template('index.html')


@main_bp.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('upload.html')

    if 'file' not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files['file']
    if not file or file.filename == '':
        flash("No selected file")
        return redirect(request.url)

    if not allowed_file(file.filename):
        flash("File type not allowed")
        return redirect(request.url)

    try:
        file_bytes = file.read()
        if not file_bytes:
            flash("Empty file")
            return redirect(request.url)

        np_img = np.frombuffer(file_bytes, np.uint8)
        image_array = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if image_array is None:
            flash("Invalid image file")
            return redirect(request.url)

        # Name + upload original
        uid = str(uuid.uuid4())
        safe_name = secure_filename(f"original_{uid}.jpg")
        original_url = upload_to_supabase(file_bytes, safe_name, file.content_type or "image/jpeg")
        if not original_url:
            flash("Error uploading original image")
            return redirect(request.url)



        # Create DB record in 'processing'
        record = ImageRecord(
            original_url=original_url,
            status='processing',
            sr_completed=False,
            yolo_completed=False,
            ocr_completed=False,
            cropping_completed=False,
            db_completed=False,
            detections_count=0,
            text_elements_count=0,
            cropped_objects_count=0,
            view_count=0,
            download_count=0,
            text_copy_count=0,
            detections=[],
            text_data=[],
            cropped_objects_urls=[]
        )
        record.save()

        # Spawn background thread with app context
        app_obj = current_app._get_current_object()
        Thread(
            target=process_in_background,
            args=(app_obj, int(record.id), file_bytes),
            daemon=True
        ).start()

        return redirect(url_for('main.processing', record_id=record.id))

    except Exception as e:
        current_app.logger.exception(f"Error initiating processing: {e}")
        flash("Error initiating image processing")
        return redirect(request.url)


@main_bp.route('/processing/<record_id>')
def processing(record_id):
    rid = _safe_int(record_id)
    if rid is None:
        flash("Invalid record id")
        return redirect(url_for('main.upload_file'))

    record = ImageRecord.get(rid)
    if not record:
        flash("Record not found")
        return redirect(url_for('main.upload_file'))

    return render_template("process.html", original_image=record.original_url, record_id=record.id)


@main_bp.route('/api/processing_status/<record_id>')
def processing_status(record_id):
    rid = _safe_int(record_id)
    if rid is None:
        return jsonify({'error': 'Invalid record id'}), 400

    record = ImageRecord.get(rid)
    if not record:
        return jsonify({'error': 'Record not found'}), 404

    data = {
        'status': record.status,
        'sr_completed': bool(record.sr_completed),
        'yolo_completed': bool(record.yolo_completed),
        'ocr_completed': bool(record.ocr_completed),
        'cropping_completed': bool(record.cropping_completed),
        'db_completed': bool(record.db_completed),
        'processed_image': record.processed_url or '',
        'detections': record.detections or [],
        'text_data': record.text_data or [],
        'cropped_objects': record.cropped_objects_urls or [],
        'record_url': url_for('main.view_record', record_id=record.id, _external=True) if record.db_completed else ''
    }
    return jsonify(data), 200


@main_bp.route('/api/process', methods=['POST'])
def api_process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        file_bytes = file.read()
        if not file_bytes:
            return jsonify({'error': 'Empty file'}), 400

        np_img = np.frombuffer(file_bytes, np.uint8)
        image_array = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if image_array is None:
            return jsonify({'error': 'Invalid image file'}), 400

        processor = get_image_processor()
        enhanced_image, detections, cropped_objs, text_data = processor.process_image(
            image_array,
            blur_sensitive=True,
            enhance_quality=True,
            extract_objects=True,
            extract_text=True
        )

        resp: Dict[str, Any] = {
            "detections": detections or [],
            "text_data": text_data or [],
            "cropped_objects_count": len(cropped_objs) if cropped_objs else 0
        }

        if request.args.get("include_images", "false").lower() == "true":
            resp["original_image"] = encode_image(image_array) or ''
            resp["processed_image"] = encode_image(enhanced_image) or ''
            resp["cropped_objects"] = [
                encode_image(obj['image_bytes']) for obj in (cropped_objs or [])
            ]

        return jsonify(resp), 200

    except Exception as e:
        current_app.logger.exception(f"API Error: {e}")
        return jsonify({'error': 'Error processing image'}), 500


@main_bp.route('/about')
def about():
    return render_template("about.html")


@main_bp.errorhandler(413)
def too_large(e):
    # Keep simple text response; customize as needed
    return "File too large", 413


@main_bp.errorhandler(500)
def internal_error(e):
    # Avoid recursion: keep minimal
    current_app.logger.exception("Unhandled 500")
    return "Internal server error", 500


@main_bp.route('/api/update_record', methods=['POST'])
def update_record():
    try:
        data = request.get_json(silent=True) or {}
        rid = _safe_int(data.get("record_id"))
        action = (data.get("action") or "").strip().lower()

        if rid is None:
            return jsonify({"status": "error", "error": "Invalid record id"}), 400
        if action not in {"download", "viewed", "copied_text"}:
            return jsonify({"status": "error", "error": "Invalid action"}), 400

        rec = ImageRecord.get(rid)
        if not rec:
            return jsonify({"status": "error", "error": "Record not found"}), 404

        if action == "download":
            rec.download_count = (rec.download_count or 0) + 1
        elif action == "viewed":
            rec.view_count = (rec.view_count or 0) + 1
        elif action == "copied_text":
            rec.text_copy_count = (rec.text_copy_count or 0) + 1

        rec.save()
        return jsonify({"status": "ok"}), 200

    except Exception as e:
        current_app.logger.exception(f"DB update failed: {e}")
        return jsonify({"status": "error"}), 500


@main_bp.route('/upload_form')
def upload_form():
    return render_template('upload.html')


@main_bp.route('/gallery')
def gallery():
    page = request.args.get('page', 1, type=int)
    limit = 20
    offset = (page - 1) * limit

    records = ImageRecord.get_all(limit=limit, offset=offset)
    total_count = ImageRecord.get_count()
    total_pages = (total_count + limit - 1) // limit if total_count else 1

    return render_template(
        'gallery.html',
        records=records,
        current_page=page,
        total_pages=total_pages
    )



@main_bp.route('/results/<record_id>')
def results(record_id):
    rid = _safe_int(record_id)
    if rid is None:
        flash("Invalid record id")
        return redirect(url_for('main.gallery'))

    record = ImageRecord.get(rid)
    if not record:
        flash("Record not found")
        return redirect(url_for('main.gallery'))

    if record.status != 'completed':
        flash("Processing not yet complete")
        return redirect(url_for('main.processing', record_id=record.id))

    return render_template(
        'results.html',
        record=record,
        original_image=record.original_url,
        processed_image=record.processed_url,
        detections=record.detections,
        text_data=record.text_data,
        cropped_objects=record.cropped_objects_urls,
        record_id=record.id
    )
@main_bp.route('/record/<record_id>')
def view_record(record_id):
    rid = _safe_int(record_id)
    if rid is None:
        flash("Invalid record id")
        return redirect(url_for('main.gallery'))

    record = ImageRecord.get(rid)
    if not record:
        flash("Record not found")
        return redirect(url_for('main.gallery'))

    # Increment view count (best-effort)
    try:
        record.view_count = (record.view_count or 0) + 1
        record.save()
    except Exception:
        current_app.logger.warning(f"Failed to increment view_count for id={rid}")

    # Simply pass the record object to the template
    return render_template('record_detail.html', record=record)

# Add this route to your blueprint
@main_bp.route('/find_similar', methods=['POST'])
def find_similar():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        image_url = data.get('image_url')
        label = data.get('label')
        confidence = data.get('confidence')
        
        if not image_url:
            return jsonify({'error': 'No image URL provided'}), 400
            
        # Find the record that contains this cropped object
        source_record = None
        all_records = ImageRecord.get_all(limit=None)  # Get all records
        
        for record in all_records:
            # Check if this record contains the query image
            if record.processed_url == image_url:
                source_record = record
                break
                
            # Check cropped objects
            for cropped_obj in record.cropped_objects_urls:
                if cropped_obj['url'] == image_url:
                    source_record = record
                    break
                    
            if source_record:
                break
                
        if not source_record:
            return jsonify({'error': 'Source image not found in records'}), 404
            
        # Find similar objects based on label and confidence
        similar_objects = []
        
        # If we have a label, find objects with the same label
        if label:
            for record in all_records:
                # Check the processed image if it matches the label
                # (This would require storing labels for the processed image, which you might not have)
                
                # Check cropped objects
                for cropped_obj in record.cropped_objects_urls:
                    if (cropped_obj.get('label') == label and 
                        cropped_obj['url'] != image_url):  # Don't include the original
                        similar_objects.append({
                            'url': cropped_obj['url'],
                            'label': cropped_obj.get('label', 'Object'),
                            'confidence': cropped_obj.get('confidence', 0),
                            'record_id': record.id
                        })
        
        # If we don't have enough similar objects by label, include other objects from the same record
        if len(similar_objects) < 3:
            for cropped_obj in source_record.cropped_objects_urls:
                if cropped_obj['url'] != image_url:  # Don't include the original
                    # Check if we already have this object
                    if not any(obj['url'] == cropped_obj['url'] for obj in similar_objects):
                        similar_objects.append({
                            'url': cropped_obj['url'],
                            'label': cropped_obj.get('label', 'Object'),
                            'confidence': cropped_obj.get('confidence', 0),
                            'record_id': source_record.id
                        })
        
        # Limit to top 5 similar objects
        similar_objects = similar_objects[:5]
        
        # Prepare response
        response = {
            'original_url': image_url,
            'label': label,
            'processed_image_url': source_record.processed_url,
            'similar_objects': similar_objects
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        current_app.logger.exception(f"Error in find_similar: {e}")
        return jsonify({'error': 'Internal server error'}), 500