from ultralytics import YOLO

from .licence_plate_detection_algorithm import PlateDetector


class YoloPlateDetector(PlateDetector):
    """
    كاشف لوحات بموديل YOLO بنفس أسلوب suliman project:
    - الكشف على crop السيارة الكامل
    - اختيار أفضل لوحة + projection / re-detection
    - OCR مشترك من PlateDetector (قص اللوحة من الفريم + batch)
    """

    _PLATE_CONF_BUCKET = 0.1

    def __init__(
        self,
        model_path,
        max_lost_frames=5,
        min_plate_confidence=0.3,
        enable_ocr=True,
        ocr_retry_interval=5,
        plate_padding=3,
    ):
        super().__init__(
            max_lost_frames=max_lost_frames,
            plate_roi_ratio=0.0,
            min_plate_confidence=min_plate_confidence,
            enable_ocr=enable_ocr,
            ocr_retry_interval=ocr_retry_interval,
            plate_padding=plate_padding,
        )
        self.model = YOLO(model_path)
        # إحصائيات للتحقق من تخفيف استدعاء الموديل
        self.stats = {
            "model_batch_calls": 0,   # كم مرة اشتغل YOLO اللوحة
            "model_cars_inferred": 0, # كم سيارة انبعتت للموديل
            "projected": 0,           # كم مرة اكتفينا بـ projection
            "redetect_queued": 0,     # كم سيارة رجعت لكشف بسبب lost
        }
        self.last_frame_stats = {
            "model_batch_calls": 0,
            "model_cars_inferred": 0,
            "projected_car_ids": [],
            "detect_car_ids": [],
            "redetect_car_ids": [],
        }

    def _best_plate(self, plates):
        """أعلى ثقة تفوز؛ عند التعادل (ضمن bucket) نأخذ الأوطى y2 — نفس suliman."""
        return max(
            plates,
            key=lambda plate: (
                round(plate[4] / self._PLATE_CONF_BUCKET),
                plate[3],
            ),
        )

    def detect_plates_for_frame(self, car_rois):
        """
        كشف اللوحات بموديل YOLO على قائمة car crops كاملة دفعة واحدة.
        ترجع: list of (x1, y1, x2, y2, conf) بإحداثيات داخل الـ crop، أو None.
        """
        if not car_rois:
            return []

        self.stats["model_batch_calls"] += 1
        self.stats["model_cars_inferred"] += len(car_rois)
        self.last_frame_stats["model_batch_calls"] += 1
        self.last_frame_stats["model_cars_inferred"] += len(car_rois)

        batched = self.model(car_rois, verbose=False)
        results = []

        for detection in batched:
            plates = detection.boxes.data.tolist() if detection.boxes is not None else []
            if not plates:
                results.append(None)
                continue

            x1, y1, x2, y2, confidence, _class_id = self._best_plate(plates)
            results.append((
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                float(confidence),
            ))

        return results

    def track_plates_for_frame(self, frame, car_dict, frame_idx, frame_size):
        """
        كشف/تتبع موقع اللوحة بأسلوب suliman ثم OCR المشترك batched.
        returns: {track_id: plate_bbox [x1,y1,x2,y2] بإحداثيات الفريم}
        """
        self.last_frame_stats = {
            "model_batch_calls": 0,
            "model_cars_inferred": 0,
            "projected_car_ids": [],
            "detect_car_ids": [],
            "redetect_car_ids": [],
        }

        results = {}
        cars_to_detect = {}

        for track_id, car_info in car_dict.items():
            car_bbox = car_info["bbox"]
            crop = car_info.get("crop")
            if crop is None or crop.size == 0:
                continue

            if track_id not in self.car_to_plate:
                cars_to_detect[track_id] = car_info
                self.last_frame_stats["detect_car_ids"].append(track_id)
                continue

            plate_id = self.car_to_plate[track_id]
            if plate_id not in self.plate_tracks:
                cars_to_detect[track_id] = car_info
                self.last_frame_stats["detect_car_ids"].append(track_id)
                continue

            plate_track = self.plate_tracks[plate_id]
            projected_bbox = self.project_bbox(car_bbox, plate_track["rel_bbox"])

            if self.plate_is_lost(plate_track, projected_bbox, frame_idx, frame_size):
                plate_track["status"] = "lost"
                plate_track["redetected"] = True
                cars_to_detect[track_id] = car_info
                self.stats["redetect_queued"] += 1
                self.last_frame_stats["redetect_car_ids"].append(track_id)
                continue

            plate_track.update({
                "bbox": projected_bbox,
                "last_seen": frame_idx,
                "status": "active",
            })
            results[track_id] = projected_bbox
            self.stats["projected"] += 1
            self.last_frame_stats["projected_car_ids"].append(track_id)

        if cars_to_detect:
            car_crops = []
            detect_order = []

            for track_id, car_info in cars_to_detect.items():
                crop = car_info["crop"]
                if crop is None or crop.size == 0:
                    continue
                car_x1, car_y1, _, _ = self._clip_bbox(frame, car_info["bbox"])
                car_crops.append(crop)
                detect_order.append((track_id, car_info, car_x1, car_y1))

            if car_crops:
                detections = self.detect_plates_for_frame(car_crops)

                for (track_id, car_info, car_x1, car_y1), plate_local in zip(
                    detect_order, detections
                ):
                    if plate_local is None:
                        continue

                    px1, py1, px2, py2, plate_conf = plate_local
                    plate_bbox = [
                        px1 + car_x1,
                        py1 + car_y1,
                        px2 + car_x1,
                        py2 + car_y1,
                    ]

                    if track_id not in self.car_to_plate:
                        plate_id = self.next_plate_id
                        self.next_plate_id += 1
                        self.car_to_plate[track_id] = plate_id
                        self.plate_tracks[plate_id] = self._new_plate_track(track_id)

                    plate_id = self.car_to_plate[track_id]
                    plate_track = self.plate_tracks[plate_id]
                    rel_bbox = self.relative_bbox(car_info["bbox"], plate_bbox)

                    plate_track.update({
                        "bbox": plate_bbox,
                        "rel_bbox": rel_bbox,
                        "confidence": plate_conf,
                        "last_seen": frame_idx,
                        "status": "active",
                        "redetected": True,
                    })
                    results[track_id] = plate_bbox

        lost_cars = []
        for track_id, plate_id in list(self.car_to_plate.items()):
            if track_id not in car_dict and plate_id in self.plate_tracks:
                if frame_idx - self.plate_tracks[plate_id]["last_seen"] > self.max_lost_frames:
                    lost_cars.append(track_id)

        for track_id in lost_cars:
            plate_id = self.car_to_plate.pop(track_id, None)
            if plate_id in self.plate_tracks:
                del self.plate_tracks[plate_id]

        self._run_ocr_for_active_plates(frame, frame_idx, frame_size, results)
        return results
