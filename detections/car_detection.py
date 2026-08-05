import cv2 as cv
import pickle
import json
from ultralytics import YOLO

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


class CarDetection:
    CLASSIFIABLE_YOLO_CLASSES = {"car", "truck"}

    def __init__(self, model_path, type_classifier=None, color_detector=None,
                 confidence_threshold=0.6):
        self.model = YOLO(model_path)
        self.type_classifier = type_classifier
        self.color_detector = color_detector          
        self.confidence_threshold = confidence_threshold

        self.tracking_log = {}

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        car_detections = [self.detect_frame(frame, idx) for idx, frame in enumerate(frames)]

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(car_detections, f)
        return car_detections

    def detect_frame(self, frame, frame_idx):
        results = self.model.track(frame, persist=True, iou=0.1, conf=0.3, verbose=False)[0]
        id_name_dict = results.names
        car_list = []
        active_track_ids = []  # 1. قائمة لتجميع الـ IDs الموجودة حالياً في الفريم

        for box in results.boxes:
            bbox = box.xyxy.tolist()[0]
            cls_id = int(box.cls.tolist()[0])
            cls_name = id_name_dict[cls_id]
            track_id = int(box.id.item()) if box.id is not None else -1
            
            # حفظ الـ ID النشط
            if track_id != -1:
                active_track_ids.append(track_id)

            # استخراج ثقة اليولو الأساسية
            yolo_conf = float(box.conf.item()) if box.conf is not None else 0.0

            if cls_name not in VEHICLE_CLASSES:
                continue

            # قيم افتراضية للوغ والتشخيص
            raw_type = "Unknown"
            type_conf = 0.0
            smoothed_type = "Unknown"
            type_vote = 0.0
            color_name = "Unknown"
            color_vote = 0.0
            display_type = cls_name

            # استدعاء تصنيف النوع
            if cls_name in self.CLASSIFIABLE_YOLO_CLASSES and self.type_classifier is not None:
                _, raw_type, type_conf, smoothed_type, type_vote = self.type_classifier.classify_and_vote(
                    frame, bbox, track_id, cls_name
                )
                display_type = smoothed_type if type_vote >= 0.5 else "Unknown"

            # استدعاء اكتشاف اللون
            if self.color_detector is not None and track_id != -1:
                color_name, color_vote = self.color_detector.get_stable_color(track_id, frame, bbox)

            # تسجيل البيانات الكاملة في الـ Log
            if track_id != -1:
                self._log_prediction(
                    track_id, frame_idx, cls_name, yolo_conf, 
                    raw_type, type_conf, smoothed_type, type_vote, 
                    color_name, color_vote
                )

            # تخزين البيانات مفصلة لتستخدمها دالة الرسم
            car_list.append({
                "bbox": bbox,
                "track_id": track_id,
                "yolo_class": cls_name,
                "yolo_conf": yolo_conf,
                "type_name": display_type,
                "type_conf": type_vote if type_vote > 0 else type_conf,
                "color_name": color_name,
                "color_conf": color_vote
            })

        # --- 2. المكان الصحيح للتنظيف (في نهاية معالجة الفريم) ---
        if self.type_classifier is not None:
            self.type_classifier.cleanup_inactive_tracks(active_track_ids)

        if self.color_detector is not None:
            self.color_detector.cleanup_inactive_tracks(active_track_ids)

        return car_list

    def _log_prediction(self, track_id, frame_idx, yolo_class, yolo_conf, raw_type, type_conf, smoothed_type, type_vote, color_name, color_vote):
        if track_id not in self.tracking_log:
            self.tracking_log[track_id] = []
            
        self.tracking_log[track_id].append({
            "frame": frame_idx,
            "yolo_class": yolo_class,                   
            "yolo_confidence": round(yolo_conf, 3),
            "raw_predicted_type": raw_type,
            "type_confidence": round(type_conf, 3) if type_conf is not None else 0.0,
            "smoothed_predicted_type": smoothed_type,
            "type_vote_ratio": round(type_vote, 3) if type_vote is not None else 0.0,
            "smoothed_color": color_name,
            "color_vote_ratio": round(color_vote, 3) if color_vote is not None else 0.0
        })

    def draw_bboxes(self, video_frames, car_detections):
        output_frames = []
        for frame, car_list in zip(video_frames, car_detections):
            for detection in car_list:
                x1, y1, x2, y2 = detection["bbox"]
                track_id = detection["track_id"]
                
                # تجهيز الأسطر بالطول فوق السيارة
                lines = []
                
                # 1. السطر الأول: معرف التتبع + صنف اليولو مع الثقة
                id_prefix = f"ID:{track_id} | " if track_id != -1 else ""
                lines.append(f"{id_prefix}{detection['yolo_class']} ({detection['yolo_conf']:.2f})")
                
                # 2. السطر الثاني: النوع مع الثقة
                lines.append(f"Type: {detection['type_name']} ({detection['type_conf']:.2f})")
                
                # 3. السطر الثالث: اللون مع الثقة
                lines.append(f"Color: {detection['color_name']} ({detection['color_conf']:.2f})")

                # حساب الموقع العمودي للبدء بالرسم فوق المستطيل (بمسافة متراصة وغير متباعدة)
                line_height = 16  # حجم المسافة بين السطر والآخر بالبكسل
                start_y = int(y1) - (len(lines) * line_height) - 4
                
                # حماية لمنع خروج النص خارج الإطار العلوي للفيديو
                if start_y < 15:
                    start_y = int(y1) + 20

                # رسم الأسطر تحت بعضها البعض
                for i, text in enumerate(lines):
                    current_y = start_y + (i * line_height)
                    cv.putText(frame, text, (int(x1), current_y),
                               cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                # رسم مستطيل السيارة حولها
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
                
            output_frames.append(frame)
        return output_frames

    def save_tracking_log(self, output_path="tracking_log.json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.tracking_log, f, ensure_ascii=False, indent=2)
        print(f"saved at {len(self.tracking_log)} different car → {output_path}")


        