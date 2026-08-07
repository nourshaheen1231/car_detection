import cv2 as cv
import pickle
import json
from ultralytics import YOLO

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


class CarDetection:
    CLASSIFIABLE_YOLO_CLASSES = {"car", "truck"}

    def __init__(
        self,
        model_path,
        type_classifier=None,
        color_detector=None,
        make_model_detector=None,
        confidence_threshold=0.6,
    ):
        self.model = YOLO(model_path)
        self.type_classifier = type_classifier
        self.color_detector = color_detector
        self.make_model_detector = make_model_detector
        self.confidence_threshold = confidence_threshold

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
        results = self.model.track(
            frame, persist=True, iou=0.1, conf=self.confidence_threshold, verbose=False
        )[0]
        id_name_dict = results.names
        car_list = []

        for box in results.boxes:
            bbox = box.xyxy.tolist()[0]
            cls_id = int(box.cls.tolist()[0])
            cls_name = id_name_dict[cls_id]
            track_id = int(box.id.item()) if box.id is not None else -1

            yolo_conf = float(box.conf.item()) if box.conf is not None else 0.0

            if cls_name not in VEHICLE_CLASSES:
                continue

            crop = self._crop(frame, bbox)

            final_type = "Unknown"
            final_conf = 0.0
            color_name = "Unknown"
            color_vote = 0.0
            make_model_name = "Unknown"
            make_model_conf = 0.0
            display_type = cls_name

            # 1. تصنيف نوع السيارة (Body Type)
            if (
                cls_name in self.CLASSIFIABLE_YOLO_CLASSES
                and self.type_classifier is not None
                and crop is not None
            ):
                (
                    final_type,
                    final_conf,
                ) = self.type_classifier.classify_and_vote(
                    crop, track_id, cls_name, frame_idx
                )

            # 2. تحديد لون السيارة
            if self.color_detector is not None and crop is not None:
                color_name, color_vote = self.color_detector.get_stable_color(
                    track_id, crop, frame_idx
                )

            # 3. تحديد الشركة والموديل (Make & Model)
            if self.make_model_detector is not None and crop is not None:
                (
                    make_model_name,
                    make_model_conf,
                ) = self.make_model_detector.get_stable_make_model(
                    track_id, crop, frame_idx
                )

            # تسجيل البيانات في السجل Log
            if track_id != -1:
                self._log_prediction(
                    track_id,
                    frame_idx,
                    cls_name,
                    yolo_conf,
                    final_type,
                    final_conf,
                    color_name,
                    color_vote,
                    make_model_name,
                    make_model_conf,
                )

            car_list.append(
                {
                    "bbox": bbox,
                    "track_id": track_id,
                    "yolo_class": cls_name,
                    "yolo_conf": yolo_conf,
                    "type_name": final_type,
                    "type_conf": final_conf,
                    "color_name": color_name,
                    "color_conf": color_vote,
                    "make_model_name": make_model_name,
                    "make_model_conf": make_model_conf,
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
            }
        )

    def draw_bboxes(self, video_frames, car_detections):
        """
        يرسم البوكسات والمعلومات فوق كل سيارة بنفس منطق test_tracking_video.py
        - خلفية ملونة خلف كل سطر
        - outline رفيع حول النص للوضوح
        - ألوان مختلفة لكل نوع معلومات
        """
        # ألوان لكل سطر (B, G, R)
        COLORS = {
            "id": (255, 255, 255),      # أبيض
            "type": (255, 200, 100),    # برتقالي فاتح
            "mmr": (100, 255, 255),     # أصفر فاتح
            "color": (150, 255, 150),   # أخضر فاتح
        }

        def _draw_label(frame, text, x, y, bg_color, font_scale=0.45, thickness=1):
            """
            نسخة من draw_label تبع test_tracking_video.py
            ترسم خلفية ملونة + outline أسود رفيع + نص داكن
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

        output_frames = []

        for frame, car_list in zip(video_frames, car_detections):
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
                    _draw_label(frame, text, int(x1), current_y, color)
                    current_y += line_height

                # نرسم البوكس تبع السيارة
                cv.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 255),  # أصفر سماوي
                    2,
                )

            output_frames.append(frame)

        return output_frames

    def save_tracking_log(self, output_path="tracking_log.json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.tracking_log, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(self.tracking_log)} vehicles to: {output_path}")