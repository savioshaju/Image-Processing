from supabase import create_client
from config import Config

supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

class ImageRecord:
    def __init__(self, original_url, processed_url=None,
                 status="pending",
                 sr_completed=False, yolo_completed=False,
                 ocr_completed=False, cropping_completed=False,
                 db_completed=False,
                 detections_count=0, text_elements_count=0, cropped_objects_count=0,
                 view_count=0, download_count=0, text_copy_count=0,
                 cropped_objects_urls=None,
                 detections=None, text_data=None,
                 record_id=None):

        self.id = record_id  # DB-assigned ID
        self.original_url = original_url
        self.processed_url = processed_url
        self.status = status

        # Ensure boolean values
        self.sr_completed = bool(sr_completed)
        self.yolo_completed = bool(yolo_completed)
        self.ocr_completed = bool(ocr_completed)
        self.cropping_completed = bool(cropping_completed)
        self.db_completed = bool(db_completed)

        self.detections_count = detections_count
        self.text_elements_count = text_elements_count
        self.cropped_objects_count = cropped_objects_count
        self.view_count = view_count
        self.download_count = download_count
        self.text_copy_count = text_copy_count

        # Ensure lists
        self.cropped_objects_urls = cropped_objects_urls if isinstance(cropped_objects_urls, list) else []
        self.detections = detections if isinstance(detections, list) else []
        self.text_data = text_data if isinstance(text_data, list) else []

    def save(self):
        data = {
            "original_url": self.original_url,
            "processed_url": self.processed_url,
            "status": self.status,
            "sr_completed": bool(self.sr_completed),
            "yolo_completed": bool(self.yolo_completed),
            "ocr_completed": bool(self.ocr_completed),
            "cropping_completed": bool(self.cropping_completed),
            "db_completed": bool(self.db_completed),
            "detections_count": self.detections_count,
            "text_elements_count": self.text_elements_count,
            "cropped_objects_count": self.cropped_objects_count,
            "view_count": self.view_count,
            "download_count": self.download_count,
            "text_copy_count": self.text_copy_count,
            "cropped_objects_urls": self.cropped_objects_urls or [],
            "detections": self.detections or [],
            "text_data": self.text_data or []
        }

        if self.id:
            supabase.table("image_records").update(data).eq("id", self.id).execute()
        else:
            result = supabase.table("image_records").insert(data).execute()
            if result.data:
                self.id = result.data[0]["id"]
            else:
                raise Exception("Failed to save record to database")

    @staticmethod
    def get(record_id):
        result = supabase.table("image_records").select("*").eq("id", record_id).execute()
        if result.data:
            row = result.data[0]
            return ImageRecord(
                record_id=row.get("id"),
                original_url=row.get("original_url"),
                processed_url=row.get("processed_url"),
                status=row.get("status", "pending"),
                sr_completed=bool(row.get("sr_completed", False)),
                yolo_completed=bool(row.get("yolo_completed", False)),
                ocr_completed=bool(row.get("ocr_completed", False)),
                cropping_completed=bool(row.get("cropping_completed", False)),
                db_completed=bool(row.get("db_completed", False)),
                detections_count=row.get("detections_count", 0),
                text_elements_count=row.get("text_elements_count", 0),
                cropped_objects_count=row.get("cropped_objects_count", 0),
                view_count=row.get("view_count", 0),
                download_count=row.get("download_count", 0),
                text_copy_count=row.get("text_copy_count", 0),
                cropped_objects_urls=row.get("cropped_objects_urls", []),
                detections=row.get("detections", []),
                text_data=row.get("text_data", [])
            )
        return None

    @staticmethod
    def get_all(limit=20, offset=0):
        result = supabase.table("image_records").select("*").order("created_at", desc=True).limit(limit).offset(offset).execute()
        records = []
        for row in result.data:
            records.append(ImageRecord(
                record_id=row.get("id"),
                original_url=row.get("original_url"),
                processed_url=row.get("processed_url"),
                status=row.get("status", "pending"),
                sr_completed=bool(row.get("sr_completed", False)),
                yolo_completed=bool(row.get("yolo_completed", False)),
                ocr_completed=bool(row.get("ocr_completed", False)),
                cropping_completed=bool(row.get("cropping_completed", False)),
                db_completed=bool(row.get("db_completed", False)),
                detections_count=row.get("detections_count", 0),
                text_elements_count=row.get("text_elements_count", 0),
                cropped_objects_count=row.get("cropped_objects_count", 0),
                view_count=row.get("view_count", 0),
                download_count=row.get("download_count", 0),
                text_copy_count=row.get("text_copy_count", 0),
                cropped_objects_urls=row.get("cropped_objects_urls", []),
                detections=row.get("detections", []),
                text_data=row.get("text_data", [])
            ))
        return records

    @staticmethod
    def get_count():
        result = supabase.table("image_records").select("id", count="exact").execute()
        return result.count
