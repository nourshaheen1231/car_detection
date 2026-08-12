import cv2 as cv
import pickle
import json
from ultralytics import YOLO

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


class CarDetection:
    CLASSIFIABLE_YOLO_CLASSES = {"car", "truck"}

    # ألوان لكل سطر (B, G, R) — على مستوى الكلاس عشان تنعمل مشاركة بين draw_frame وdraw_bboxes
    _DRAW_COLORS = {
        "id": (255, 255, 255),      # أبيض
        "type": (255, 200, 100),    # برتقالي فاتح
        "mmr": (100, 255, 255),     # أصفر فاتح
        "color": (150, 255, 150),   # أخضر فاتح
    }

    def __init__(
        self,
        model_path,
        type_classifier=None,
        color_detector=None,
        make_model_detector=None,
        plate_detector=None, 
        confidence_threshold=0.6,
        plate_roi_ratio=0.45, 
    ):
        self.model = YOLO(model_path)
        self.type_classifier = type_classifier
        self.color_detector = color_detector
        self.make_model_detector = make_model_detector
        self.plate_detector = plate_detector
        self.confidence_threshold = confidence_threshold
        self.plate_roi_ratio = plate_roi_ratio

        self.tracking_log = {}

    def _crop(self, frame, bbox):
        """نسخة موحدة وآمنة من الـ crop"""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    def _clip_bbox(self, frame, bbox):
            """يرجع إحداثيات الـ bbox بعد قصّها لتطابق حدود الفريم (نفس منطق _crop لكن بيرجع الإحداثيات)."""
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            return x1, y1, x2, y2

    def _crop_plate_roi(self, car_crop):

            if car_crop is None or car_crop.size == 0:
                return None, 0

            h, w = car_crop.shape[:2]
            y_offset = int(h * self.plate_roi_ratio)
            return car_crop[y_offset:h, 0:w], y_offset



    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        car_detections = [
            self.detect_frame(frame, idx) for idx, frame in enumerate(frames)
        ]

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(car_detections, f)
        return car_detections

    def detect_frame(self, frame, frame_idx):
        """
        معالجة الإطار الواحد بشكل مجمع (Batched) لكل الموديلات.
        """
        frame_height, frame_width = frame.shape[:2]

        # تتبع السيارات: ByteTrack الافتراضي (مش BoT-SORT) — ثابت عبر persist=True
        results = self.model.track(
            frame, persist=True, iou=0.1, conf=self.confidence_threshold, verbose=False
        )[0]
        id_name_dict = results.names
        car_list = []

        # =====================================================
        # المرحلة 1: تجميع البيانات الأساسية لكل السيارات
        # =====================================================
        car_crops = {}      # track_id -> bbox (للباتشينج)
        car_metadata = {}   # track_id -> metadata كاملة


        for box in results.boxes:
            bbox = box.xyxy.tolist()[0]
            cls_id = int(box.cls.tolist()[0])
            cls_name = id_name_dict[cls_id]
            track_id = int(box.id.item()) if box.id is not None else -1

            yolo_conf = float(box.conf.item()) if box.conf is not None else 0.0

            if cls_name not in VEHICLE_CLASSES:
                continue

            crop = self._crop(frame, bbox)

            if track_id != -1 and crop is not None:
                car_crops[track_id] = bbox
                car_metadata[track_id] = {
                    'bbox': bbox,
                    'cls_name': cls_name,
                    'yolo_conf': yolo_conf,
                    'crop': crop,
                }



        # =====================================================
        # المرحلة 2: استدعاء الموديلات بشكل مجمع (Batched)
        # =====================================================

        # 1. اكتشاف اللوحات (Smart Tracking: projection + re-detection + lost tracks)
        # بدل ما نكشف اللوحة بكل فريم، نتبعها بـ projection ونعيد الكشف بس لما تضيع
        plates = {}
        if self.plate_detector is not None and car_crops:
            plates = self.plate_detector.track_plates_for_frame(
                frame, car_metadata, frame_idx, (frame_width, frame_height)
            )

        # 2. تحديد لون السيارة (Batched)
        colors = {}
        if self.color_detector is not None and car_crops:
            colors = self.color_detector.get_stable_colors_for_frame(
                frame, car_crops, frame_idx
            )

        # 3. تصنيف نوع جسم السيارة (Batched)
        types = {}
        if self.type_classifier is not None and car_crops:
            cls_name_dict = {tid: car_metadata[tid]['cls_name'] for tid in car_crops}
            types = self.type_classifier.classify_and_vote_for_frame(
                frame, car_crops, cls_name_dict, frame_idx
            )

        # 4. تحديد الشركة والموديل (Batched)
        mmrs = {}
        if self.make_model_detector is not None and car_crops:
            mmrs = self.make_model_detector.get_stable_make_models_for_frame(
                frame, car_crops, frame_idx
            )



        # =====================================================
        # المرحلة 3: بناء قائمة النتائج النهائية
        # =====================================================
        for track_id, meta in car_metadata.items():
            color_name, color_vote = colors.get(track_id, ("Unknown", 0.0))
            final_type, final_conf = types.get(track_id, ("Unknown", 0.0))
            make_model_name, make_model_conf = mmrs.get(track_id, ("Unknown", 0.0))
            plate_bbox = plates.get(track_id, None)

            # الربط سيارة ↔ لوحة (مثل suliman: vehicle_to_plate)
            plate_track_id = None
            plate_text = None
            plate_text_conf = None
            if self.plate_detector is not None:
                plate_track_id = self.plate_detector.car_to_plate.get(track_id)
                if plate_track_id is not None:
                    pt = self.plate_detector.plate_tracks.get(plate_track_id, {})
                    plate_text = pt.get("text")
                    plate_text_conf = pt.get("text_conf")

            # تسجيل البيانات في السجل Log
            self._log_prediction(
                track_id,
                frame_idx,
                meta['cls_name'],
                meta['yolo_conf'],
                final_type,
                final_conf,
                color_name,
                color_vote,
                make_model_name,
                make_model_conf,
                plate_bbox,
                plate_track_id,
                plate_text,
                plate_text_conf,
            )

            car_list.append(
                {
                    "bbox": meta['bbox'],
                    "track_id": track_id,
                    "yolo_class": meta['cls_name'],
                    "yolo_conf": meta['yolo_conf'],
                    "type_name": final_type,
                    "type_conf": final_conf,
                    "color_name": color_name,
                    "color_conf": color_vote,
                    "make_model_name": make_model_name,
                    "make_model_conf": make_model_conf,
                    "plate_bbox": plate_bbox,
                    "plate_track_id": plate_track_id,
                    "plate_text": plate_text,
                    "plate_text_conf": plate_text_conf,
                }
            )

        # تنظيف الكاش للسيارات الغائبة
        if self.type_classifier is not None:
            self.type_classifier.cleanup_inactive_tracks(frame_idx)

        if self.color_detector is not None:
            self.color_detector.cleanup_inactive_tracks(frame_idx)

        if self.make_model_detector is not None:
            self.make_model_detector.cleanup_inactive_tracks(frame_idx)

        return car_list

    def _log_prediction(
        self,
        track_id,
        frame_idx,
        yolo_class,
        yolo_conf,
        smoothed_type,
        type_conf,
        color_name,
        color_vote,
        make_model_name,
        make_model_conf,
        plate_bbox=None,
        plate_track_id=None,
        plate_text=None,
        plate_text_conf=None,
    ):
        if track_id not in self.tracking_log:
            self.tracking_log[track_id] = []

        self.tracking_log[track_id].append(
            {
                "frame": frame_idx,
                "yolo_class": yolo_class,
                "yolo_confidence": round(yolo_conf, 3)
                if yolo_conf is not None
                else 0.0,
                "predicted_type": smoothed_type,
                "type_confidence": round(type_conf, 3)
                if type_conf is not None
                else 0.0,
                "smoothed_color": color_name,
                "color_vote_ratio": round(color_vote, 3)
                if color_vote is not None
                else 0.0,
                "make_model": make_model_name,
                "make_model_confidence": round(make_model_conf, 3)
                if make_model_conf is not None
                else 0.0,
                "plate_bbox": plate_bbox,
                "plate_track_id": plate_track_id,
                "plate_text": plate_text,
                "plate_text_conf": round(plate_text_conf, 3)
                if plate_text_conf is not None
                else None,
            }
        )

    # =====================================================
    # دوال الرسم — النسخة الجديدة (streaming) + القديمة (batch)
    # =====================================================
    @staticmethod
    def _draw_label(frame, text, x, y, bg_color, font_scale=0.45, thickness=1):
        """
        ترسم خلفية ملونة + outline أسود رفيع + نص داكن.
        نسخة static method مشتركة بين draw_frame و draw_bboxes.
        """
        (text_w, text_h), _ = cv.getTextSize(
            text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        # نتأكد ما يطلع النص فوق الشاشة
        y = max(y, text_h + 6)

        pad = 4
        # 1. الخلفية الملونة
        cv.rectangle(
            frame,
            (x, y - text_h - pad),
            (x + text_w + pad * 2, y + pad),
            bg_color,
            -1,  # fill
        )

        # 2. outline أسود (رفيع) للوضوح
        cv.putText(
            frame, text, (x + pad, y - 2),
            cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv.LINE_AA
        )
        # 3. النص الأصلي
        cv.putText(
            frame, text, (x + pad, y - 2),
            cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv.LINE_AA
        )

    def draw_frame(self, frame, car_list):
        """
        يرسم البوكسات والمعلومات فوق فريم واحد بس (بالمكان، in-place) ويرجّعه.
        هاي هي الدالة يلي بتستخدمها المعالجة الـ streaming (فريم فريم) بدل تجميع
        كل الفريمات بالذاكرة أول ما بتخلص المعالجة.
        """
        COLORS = self._DRAW_COLORS

        for detection in car_list:
            x1, y1, x2, y2 = detection["bbox"]
            track_id = detection["track_id"]

            # نجهز الأسطر يلي بدنا نرسمن
            lines = []

            # السطر الأول: ID + صنف YOLO
            id_prefix = f"ID:{track_id} | " if track_id != -1 else ""
            lines.append((f"{id_prefix}{detection['yolo_class']} ({detection['yolo_conf']:.2f})", COLORS["id"]))

            # السطر الثاني: النوع (Type)
            lines.append((f"Type: {detection['type_name']} ({detection['type_conf']:.2f})", COLORS["type"]))

            # السطر الثالث: Make & Model (بس إذا مش Unknown)
            if detection.get("make_model_name") and detection["make_model_name"] != "Unknown":
                lines.append((f"MMR: {detection['make_model_name']} ({detection['make_model_conf']:.2f})", COLORS["mmr"]))

            # السطر الرابع: اللون
            lines.append((f"Color: {detection['color_name']} ({detection['color_conf']:.2f})", COLORS["color"]))

            # نحسب ارتفاع كل سطر
            line_height = 18
            total_height = len(lines) * line_height + 4

            # نحدد موقع البداية (فوق البوكس)
            start_y = int(y1) - total_height

            # إذا ما فيش مساحة فوق البوكس، نرسم تحتو
            if start_y < 10:
                start_y = int(y2) + 18

            # نرسم كل سطر
            current_y = start_y
            for text, color in lines:
                self._draw_label(frame, text, int(x1), current_y, color)
                current_y += line_height

            # نرسم البوكس تبع السيارة
            cv.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 255),  # أصفر سماوي
                2,
            )

            # =====================================================
            # Plate Detection Drawing
            # =====================================================
            if detection.get("plate_bbox") is not None:
                px1, py1, px2, py2 = [int(v) for v in detection["plate_bbox"]]
                cv.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                plate_text = detection.get("plate_text")
                if plate_text:
                    plate_label = str(plate_text)
                else:
                    plate_id = detection.get("plate_track_id")
                    plate_label = f"Plate:{plate_id}" if plate_id is not None else "Plate"
                self._draw_label(frame, plate_label, px1, py1 - 4, (0, 0, 255))

        return frame

    def draw_bboxes(self, video_frames, car_detections):
        """
        نسخة الدفعة (batch) القديمة — بتاخد كل الفريمات دفعة وحدة وبترجع كل شي مرسوم.
        محفوظة لأي كود قديم عم يعتمد عليها. تحتها بتستخدم draw_frame لكل فريم
        (نفس النتيجة بالضبط متل قبل، بس بدون تكرار الكود).
        """
        output_frames = []
        for frame, car_list in zip(video_frames, car_detections):
            self.draw_frame(frame, car_list)
            output_frames.append(frame)
        return output_frames

    def save_tracking_log(self, output_path="tracking_log.json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.tracking_log, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(self.tracking_log)} vehicles to: {output_path}")