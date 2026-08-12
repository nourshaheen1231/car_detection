import cv2 as cv
import pickle
import numpy as np
import os

from .plate_ocr import read_license_plates


class PlateDetector:

    def __init__(
        self,
        max_lost_frames=5,
        plate_roi_ratio=0.45,
        min_plate_confidence=0.3,
        enable_ocr=True,
        ocr_retry_interval=5,
        plate_padding=3,
    ):
        """
        تهيئة كاشف اللوحة (خوارزمية CV) مع دعم التتبع الذكي (Projection + Re-detection).

        Args:
            max_lost_frames: عدد الفريمات الأقصى قبل اعتبار اللوحة مفقودة
            plate_roi_ratio: نسبة القص لأخذ الجزء السفلي من السيارة
            min_plate_confidence: أدنى ثقة مقبولة لمسار اللوحة
            enable_ocr: تشغيل قراءة نص اللوحة (PaddleOCR) بعد تثبيت الموقع
            ocr_retry_interval: عدد الفريمات بين محاولات OCR عند فشل القراءة
            plate_padding: هامش قص اللوحة من الفريم قبل OCR
        """
        self.max_lost_frames = max_lost_frames
        self.plate_roi_ratio = plate_roi_ratio
        self.min_plate_confidence = min_plate_confidence
        self.enable_ocr = enable_ocr
        self.ocr_retry_interval = ocr_retry_interval
        self.plate_padding = plate_padding

        # تتبع اللوحات: {plate_track_id: {rel_bbox, confidence, last_seen, status, ...}}
        self.plate_tracks = {}
        # ربط السيارة باللوحة: {car_track_id: plate_track_id}
        self.car_to_plate = {}
        self.next_plate_id = 101

        # إحصائيات مرئية للتيست: CV detector vs projection
        self.last_frame_stats = {
            "cv_detect_calls": 0,
            "cv_cars_inferred": 0,
            "projected_car_ids": [],
            "detect_car_ids": [],
            "redetect_car_ids": [],
        }
    def _new_plate_track(self, vehicle_id):
        return {
            "vehicle_id": vehicle_id,
            "ocr_done": False,
            "ocr_attempts": 0,
            "last_ocr_frame": -10**9,
            "redetected": False,
            "text": None,
            "text_conf": None,
        }

    # =========================================================
    # دوال التتبع الأساسية (من شغل الزميل الأول)
    # =========================================================
    def relative_bbox(self, vehicle_bbox, plate_bbox):
        """Where the plate sits inside its vehicle box, as 0..1 ratios."""
        xcar1, ycar1, xcar2, ycar2 = vehicle_bbox
        car_width = max(xcar2 - xcar1, 1)
        car_height = max(ycar2 - ycar1, 1)
        x1, y1, x2, y2 = plate_bbox
        return [(x1 - xcar1) / car_width, (y1 - ycar1) / car_height,
                (x2 - xcar1) / car_width, (y2 - ycar1) / car_height]

    def project_bbox(self, vehicle_bbox, rel_bbox):
        """Put a relative plate box back on a vehicle box, the plate is bolted to
        the car so it follows the vehicle track without running the detector."""
        xcar1, ycar1, xcar2, ycar2 = vehicle_bbox
        car_width = xcar2 - xcar1
        car_height = ycar2 - ycar1
        rx1, ry1, rx2, ry2 = rel_bbox
        return [xcar1 + rx1 * car_width, ycar1 + ry1 * car_height,
                xcar1 + rx2 * car_width, ycar1 + ry2 * car_height]

    def plate_is_lost(self, plate_track, plate_bbox, frame_num, frame_size):
        """A projected plate box stops being trustworthy when it collapses, leaves
        the frame, was anchored on a weak detection or went unseen for too long."""
        frame_width, frame_height = frame_size
        x1, y1, x2, y2 = plate_bbox

        # the projection is unusable, only the detector can put the track back
        if x2 - x1 < 1 or y2 - y1 < 1:
            return True

        if x2 <= 0 or y2 <= 0 or x1 >= frame_width or y1 >= frame_height:
            return True

        if frame_num - plate_track['last_seen'] > self.max_lost_frames:
            return True

        # a weak anchor is worth one detector run, after that the ocr carries on
        # alone instead of paying for the detector on every single frame
        return plate_track['confidence'] < self.min_plate_confidence and not plate_track.get('redetected', False)

    # =========================================================
    # BATCHED DETECTION (من المهمة الأولى)
    # =========================================================
    def detect_plates_for_frame(self, car_rois):
        """
        تعالج قائمة من car ROIs دفعة واحدة لاكتشاف اللوحات (خوارزمية CV).
        car_rois: list of numpy arrays (car crops)
        ترجع: list of tuples (x, y, w, h, conf) أو None لكل ROI
        """
        results = []
        for car_roi in car_rois:
            plate_box = self.detect_plate_location_dynamic_guassanian(car_roi)
            if plate_box is None:
                results.append(None)
            else:
                x, y, w, h = plate_box
                results.append((x, y, w, h, 0.5))
        return results

    # =========================================================
    # SMART PLATE TRACKING — التتبع الذكي للوحة
    # =========================================================
    def track_plates_for_frame(self, frame, car_dict, frame_idx, frame_size):
        """
        التتبع الذكي للوحات: projection + re-detection + lost tracks.

        car_dict: {track_id: {'bbox': [...], 'crop': np.array, 'cls_name': str}, ...}
        returns: {track_id: plate_bbox أو None}
        """
        self.last_frame_stats = {
            "cv_detect_calls": 0,
            "cv_cars_inferred": 0,
            "projected_car_ids": [],
            "detect_car_ids": [],
            "redetect_car_ids": [],
        }

        results = {}
        cars_to_detect = {}  # سيارات بحاجة لكشف جديد/إعادة

        # ---------------------------------------------------------
        # المرحلة 1: projection للوحات الموجودة (اللوحة "مسمرة" على السيارة)
        # ---------------------------------------------------------
        for track_id, car_info in car_dict.items():
            car_bbox = car_info['bbox']

            # سيارة جديدة — ما عندها مسار لوحة
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
            projected_bbox = self.project_bbox(car_bbox, plate_track['rel_bbox'])

            # إذا الـ projection ما بيطلع صحيح — نرجع نشغل الـ detector
            if self.plate_is_lost(plate_track, projected_bbox, frame_idx, frame_size):
                plate_track['status'] = 'lost'
                plate_track['redetected'] = True
                cars_to_detect[track_id] = car_info
                self.last_frame_stats["redetect_car_ids"].append(track_id)
                continue

            # Projection ناجح — حدث المسار بدون ما تشغل الـ detector
            plate_track.update({
                'bbox': projected_bbox,
                'last_seen': frame_idx,
                'status': 'active'
            })
            results[track_id] = projected_bbox
            self.last_frame_stats["projected_car_ids"].append(track_id)

        # ---------------------------------------------------------
        # المرحلة 2: كشف اللوحات للسيارات المحتاجة فقط (re-detection)
        # ---------------------------------------------------------
        if cars_to_detect:
            plate_rois = []
            detect_order = []

            for track_id, car_info in cars_to_detect.items():
                crop = car_info['crop']
                plate_roi, y_offset = self._crop_plate_roi(crop)
                if plate_roi is not None:
                    plate_rois.append(plate_roi)
                    detect_order.append((track_id, car_info, y_offset))

            if plate_rois:
                self.last_frame_stats["cv_detect_calls"] = 1
                self.last_frame_stats["cv_cars_inferred"] = len(plate_rois)
                detections = self.detect_plates_for_frame(plate_rois)

                for (track_id, car_info, y_offset), plate_local_box in zip(detect_order, detections):
                    if plate_local_box is None:
                        continue

                    px, py, pw, ph, plate_conf = plate_local_box
                    x1_car, y1_car, _, _ = self._clip_bbox(frame, car_info['bbox'])
                    plate_bbox = [
                        x1_car + px,
                        y1_car + y_offset + py,
                        x1_car + px + pw,
                        y1_car + y_offset + py + ph,
                    ]

                    # إنشاء مسار لوحة جديد إذا لسا ما عندو
                    if track_id not in self.car_to_plate:
                        plate_id = self.next_plate_id
                        self.next_plate_id += 1
                        self.car_to_plate[track_id] = plate_id
                        self.plate_tracks[plate_id] = self._new_plate_track(track_id)

                    plate_id = self.car_to_plate[track_id]
                    plate_track = self.plate_tracks[plate_id]

                    # حفظ الموقع النسبي (rel_bbox) — هاد سر التتبع
                    rel_bbox = self.relative_bbox(car_info['bbox'], plate_bbox)

                    plate_track.update({
                        'bbox': plate_bbox,
                        'rel_bbox': rel_bbox,
                        'confidence': plate_conf,
                        'last_seen': frame_idx,
                        'status': 'active',
                        'redetected': True
                    })

                    results[track_id] = plate_bbox

        # ---------------------------------------------------------
        # المرحلة 3: تنظيف المسارات الميتة (lost tracks)
        # ---------------------------------------------------------
        lost_cars = []
        for track_id, plate_id in list(self.car_to_plate.items()):
            if track_id not in car_dict:
                # السيارة غابت — تحقق من مدة الغياب
                if plate_id in self.plate_tracks:
                    if frame_idx - self.plate_tracks[plate_id]['last_seen'] > self.max_lost_frames:
                        lost_cars.append(track_id)

        for track_id in lost_cars:
            plate_id = self.car_to_plate.pop(track_id, None)
            if plate_id in self.plate_tracks:
                del self.plate_tracks[plate_id]

        # ---------------------------------------------------------
        # المرحلة 4: OCR مشترك (قص اللوحة من الفريم + batch واحد)
        # ---------------------------------------------------------
        self._run_ocr_for_active_plates(frame, frame_idx, frame_size, results)

        return results

    # =========================================================
    # SHARED OCR — قص اللوحة من الفريم ثم قراءة batched
    # =========================================================
    def _should_run_ocr(self, plate_track, frame_idx):
        if not self.enable_ocr:
            return False
        if plate_track.get("ocr_done", False):
            return False
        last_ocr = plate_track.get("last_ocr_frame", -10**9)
        return (frame_idx - last_ocr) >= self.ocr_retry_interval

    def _crop_plate_from_frame(self, frame, plate_bbox, frame_size):
        frame_width, frame_height = frame_size
        x1 = max(0, int(plate_bbox[0]) - self.plate_padding)
        y1 = max(0, int(plate_bbox[1]) - self.plate_padding)
        x2 = min(frame_width, int(plate_bbox[2]) + self.plate_padding)
        y2 = min(frame_height, int(plate_bbox[3]) + self.plate_padding)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None
        return crop

    def _run_ocr_for_active_plates(self, frame, frame_idx, frame_size, active_results):
        """
        active_results: {car_track_id: plate_bbox} للوحات الظاهرة هذا الفريم.
        يجمع crops المستحقة ثم يشغّل PaddleOCR مرة واحدة للفريم.
        """
        if not self.enable_ocr or not active_results:
            return

        plate_crops = []
        plate_ids = []

        for track_id, plate_bbox in active_results.items():
            plate_id = self.car_to_plate.get(track_id)
            if plate_id is None or plate_id not in self.plate_tracks:
                continue

            plate_track = self.plate_tracks[plate_id]
            if not self._should_run_ocr(plate_track, frame_idx):
                continue

            crop = self._crop_plate_from_frame(frame, plate_bbox, frame_size)
            if crop is None:
                continue

            plate_crops.append(crop)
            plate_ids.append(plate_id)

        if not plate_crops:
            return

        ocr_results = read_license_plates(plate_crops)

        for plate_id, (text, text_conf) in zip(plate_ids, ocr_results):
            plate_track = self.plate_tracks[plate_id]
            plate_track["ocr_attempts"] = plate_track.get("ocr_attempts", 0) + 1
            plate_track["last_ocr_frame"] = frame_idx

            if text is None:
                continue

            plate_track["text"] = text
            plate_track["text_conf"] = text_conf
            plate_track["ocr_done"] = True

    def reset(self):
        """تفريغ كلي لمسارات اللوحات عند إعادة تشغيل الفيديو"""
        self.plate_tracks.clear()
        self.car_to_plate.clear()
        self.next_plate_id = 101

    # =========================================================
    # دوال مساعدة للتتبع
    # =========================================================
    def _crop_plate_roi(self, car_crop):
        """يقتطع الجزء السفلي من صورة السيارة (حيث تقع اللوحة عادةً)."""
        if car_crop is None or car_crop.size == 0:
            return None, 0
        h, w = car_crop.shape[:2]
        y_offset = int(h * self.plate_roi_ratio)
        return car_crop[y_offset:h, 0:w], y_offset

    def _clip_bbox(self, frame, bbox):
        """يرجع إحداثيات الـ bbox بعد قصّها لتطابق حدود الفريم."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return x1, y1, x2, y2

    def _calculate_projection_profile_transitions(self, edge_roi: np.ndarray, w: int, h: int) -> int:
        """
        يحسب عدد الانتقالات بناءً على الإسقاط العمودي لشريحة وسطية (متين ضد الانزياح).
        """
        strip_y1, strip_y2 = int(h * 0.33), int(h * 0.66)
        if strip_y2 <= strip_y1:
            return 0 # حماية ضد الصناديق الصغيرة جداً

        mid_strip = edge_roi[strip_y1:strip_y2, :]
        strip_h = strip_y2 - strip_y1

        # جمع البكسلات البيضاء عمودياً لكل عمود في الشريحة
        col_sums = np.sum(mid_strip > 0, axis=0)

        # يعتبر العمود جزءاً من حرف إذا كان 20% على الأقل من بكسلاته بيضاء
        col_binary = (col_sums > (strip_h * 0.2)).astype(np.uint8)

        # حساب عدد الانتقالات
        transitions = np.sum(col_binary[:-1] != col_binary[1:])
        return transitions

    def detect_plate_location_dynamic_guassanian(self, car_roi: np.ndarray) -> tuple | None:
        """
        الوظيفة: اكتشاف موقع اللوحة في الوقت الفعلي (Video Stream) بمعمارية نظيفة.
        تتضمن بوابات رفض صارمة، دوال تقييم غاوسية، وتتبع تشخيصي للأخطاء.
        """
        if car_roi is None or car_roi.size == 0:
            return None

        h_car, w_car = car_roi.shape[:2]
        car_area = w_car * h_car
        roi_center_x = w_car / 2.0

        # تعريف أحجام الكيرنل ديناميكياً
        kernel_w = max(3, int(w_car * 0.035))
        kernel_h = max(2, int(h_car * 0.02))

        min_plate_area = car_area * 0.005
        max_plate_area = car_area * 0.15

        gray = cv.cvtColor(car_roi, cv.COLOR_BGR2GRAY)

        # ---------------------------------------------------------
        # Pipeline A: Adaptive Thresholding (مسار استخراج الكونتورات)
        # ---------------------------------------------------------
        blurred_a = cv.GaussianBlur(gray, (3, 3), 0)
        char_binary = cv.adaptiveThreshold(
            blurred_a, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv.THRESH_BINARY_INV, 15, 7
        )

        kernel_merge = cv.getStructuringElement(cv.MORPH_RECT, (kernel_w + 5, 2))
        merged_strip = cv.morphologyEx(char_binary, cv.MORPH_CLOSE, kernel_merge)

        kernel_clean = cv.getStructuringElement(cv.MORPH_RECT, (1, max(3, kernel_h // 2)))
        closed_contours_mask = cv.morphologyEx(merged_strip, cv.MORPH_OPEN, kernel_clean)

        # ---------------------------------------------------------
        # Pipeline B: Texture Extraction (مسار استخراج الملمس للتقييم)
        # ---------------------------------------------------------
        gray_smooth_b = cv.GaussianBlur(gray, (5, 5), 0)
        tophat_kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 5))
        tophat = cv.morphologyEx(gray_smooth_b, cv.MORPH_TOPHAT, tophat_kernel)

        blurred_b = cv.GaussianBlur(tophat, (5, 5), 0)
        sobel_x = cv.Sobel(blurred_b, cv.CV_16S, dx=1, dy=0, ksize=3)
        abs_sobel_x = cv.convertScaleAbs(sobel_x)

        _, threshed = cv.threshold(abs_sobel_x, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # ---------------------------------------------------------
        # Logic Gates & Feature Extraction
        # ---------------------------------------------------------
        contours, _ = cv.findContours(closed_contours_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        candidates = []

        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            box_area = w * h

            # 1. Hard Geometry Gates (بوابات الرفض الهندسية)
            if h < 3 or w < 3 or box_area == 0:
                continue
            if box_area < min_plate_area or box_area > max_plate_area:
                continue

            actual_contour_area = cv.contourArea(contour)
            extent = float(actual_contour_area) / box_area
            aspect_ratio = float(w) / h

            if aspect_ratio < 1.2 or aspect_ratio > 6.5:
                continue
            if extent < 0.35:
                continue

            # 2. Hard Density Gates (بوابات الرفض للكثافة)
            edge_roi = threshed[y:y+h, x:x+w]
            white_pixels = cv.countNonZero(edge_roi)
            density_ratio = white_pixels / box_area

            # قتل الصناديق الناعمة تماماً كالأسفلت وظلال السيارات
            if white_pixels < 15 or density_ratio < 0.08:
                continue 

            # ---------------------------------------------------------
            # Complex Scoring Functions (دوال التقييم الرياضية المتقدمة)
            # ---------------------------------------------------------

            # أ. الميزات الهندسية (Shape)
            ar_score = np.exp(-((aspect_ratio - 3.5) ** 2) / 2.0)
            s_geo = ar_score * extent  

            # ب. الموقع المكاني (Spatial X & Y)
            box_center_x = x + (w / 2.0)
            distance_from_center_x = abs(roi_center_x - box_center_x)
            s_hx = max(0, 1.0 - (distance_from_center_x / roi_center_x)) 

            box_center_y = y + (h / 2.0)
            y_norm = box_center_y / h_car 
            s_vy = np.exp(-((y_norm - 0.65) ** 2) / 0.15) # دالة ذروة واسعة النطاق

            # ج. كثافة الملمس (Texture Density)
            s_density = np.exp(-((density_ratio - 0.30) ** 2) / 0.04) # دالة ذروة تستهدف 30%

            # د. الانتظام النصي (Text Pattern Consistency)
            transitions = self._calculate_projection_profile_transitions(edge_roi, w, h)

            if 8 <= transitions <= 20:
                pattern_score = 1.0
            elif 5 <= transitions < 8 or 20 < transitions <= 28:
                pattern_score = 0.5
            else:
                pattern_score = 0.01 # أرضية منخفضة لسحق المربعات الخاطئة (كالشعارات)

            # هـ. الدمج الجدائي (Multiplicative Fusion)
            final_score = (s_geo * 1.5) * (s_hx * s_vy) * s_density * pattern_score

            # ---------------------------------------------------------
            # Diagnostic Telemetry (نظام التشخيص والمراقبة)
            # ---------------------------------------------------------
            print(f"[CANDIDATE BOX] x:{x}, y:{y}, w:{w}, h:{h}")
            print(f" ├─ Shape (s_geo)    : {s_geo:.3f} | ar:{aspect_ratio:.2f}, extent:{extent:.2f}")
            print(f" ├─ X-Pos (s_hx)     : {s_hx:.3f} | dist_from_center:{distance_from_center_x:.1f}")
            print(f" ├─ Y-Pos (s_vy)     : {s_vy:.3f} | y_norm:{y_norm:.2f}")
            print(f" ├─ Density (s_dens) : {s_density:.3f} | raw_ratio:{density_ratio:.3f}")
            print(f" ├─ Pattern (s_pat)  : {pattern_score:.3f} | transitions:{transitions}")
            print(f" └─ FINAL SCORE      : {final_score:.4f}\n")

            candidates.append({
                'box': (x, y, w, h),
                'score': final_score
            })

        # إرجاع أفضل مرشح إن وُجد
        if not candidates:
            return None

        best_candidate = sorted(candidates, key=lambda k: k['score'], reverse=True)[0]
        return best_candidate['box']



    def detect_plate_location_dynamic(self, car_roi): 
        """
        Executes a dual-pipeline architecture (Adaptive for contours, Sobel for texture).
        """
        if car_roi is None or car_roi.size == 0:
            return None

        h_car, w_car = car_roi.shape[:2]
        car_area = w_car * h_car

        kernel_w = max(3, int(w_car * 0.035))
        kernel_h = max(2, int(h_car * 0.02))

        min_plate_area = car_area * 0.005
        max_plate_area = car_area * 0.15

        gray = cv.cvtColor(car_roi, cv.COLOR_BGR2GRAY)

        # ---------------------------------------------------------
        # Pipeline A: Adaptive Thresholding for Contours (Shape Detection)
        # ---------------------------------------------------------
        blurred_a = cv.GaussianBlur(gray, (3, 3), 0)
        char_binary = cv.adaptiveThreshold(
            blurred_a, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv.THRESH_BINARY_INV, 15, 7
        )

        kernel_merge = cv.getStructuringElement(cv.MORPH_RECT, (kernel_w + 5, 2))
        merged_strip = cv.morphologyEx(char_binary, cv.MORPH_CLOSE, kernel_merge)

        kernel_clean = cv.getStructuringElement(cv.MORPH_RECT, (1, max(3, kernel_h // 2)))
        closed_contours_mask = cv.morphologyEx(merged_strip, cv.MORPH_OPEN, kernel_clean)

        # ---------------------------------------------------------
        # Pipeline B: Texture Extraction for Scoring (Edge Density)
        # ---------------------------------------------------------
        gray_smooth_b = cv.GaussianBlur(gray, (5, 5), 0)
        tophat_kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 5))
        tophat = cv.morphologyEx(gray_smooth_b, cv.MORPH_TOPHAT, tophat_kernel)

        blurred_b = cv.GaussianBlur(tophat, (5, 5), 0)
        sobel_x = cv.Sobel(blurred_b, cv.CV_16S, dx=1, dy=0, ksize=3)
        abs_sobel_x = cv.convertScaleAbs(sobel_x)

        _, threshed = cv.threshold(abs_sobel_x, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # ---------------------------------------------------------
        # Logic Gates: Filtering & Extraction
        # ---------------------------------------------------------
        contours, _ = cv.findContours(closed_contours_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        roi_center_x = w_car / 2.0
        candidates = []

        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            box_area = w * h

            # Hard Filters
            if h < 3 or w < 3 or box_area == 0:
                continue

            if box_area < min_plate_area or box_area > max_plate_area:
                continue

            actual_contour_area = cv.contourArea(contour)
            extent = float(actual_contour_area) / box_area
            aspect_ratio = float(w) / h

            if aspect_ratio < 1.2 or aspect_ratio > 6.5:
                continue
            if extent < 0.35:
                continue

            # ---------------------------------------------------------
            # Complex Scoring Function
            # ---------------------------------------------------------
            # 1. Geometry Features
            ar_score = np.exp(-((aspect_ratio - 3.5) ** 2) / 2.0)
            s_geo = ar_score * extent  

            # 2. Spatial Features
            box_center_x = x + (w / 2.0)
            distance_from_center_x = abs(roi_center_x - box_center_x)
            s_hx = max(0, 1.0 - (distance_from_center_x / roi_center_x)) 

            box_center_y = y + (h / 2.0)
            s_vy = box_center_y / h_car


            # 3. Texture & Pattern Features (from Pipeline B)
            edge_roi = threshed[y:y+h, x:x+w]

            white_pixels = cv.countNonZero(edge_roi)
            density_ratio = white_pixels / box_area
            s_density = min(1.0, density_ratio * 2.5) 

            # Zero-Crossings / Textual Consistency Check
            mid_row = edge_roi[h // 2, :] 
            transitions = np.sum(mid_row[:-1] != mid_row[1:])

            pattern_score = 0.1
            if 8 <= transitions <= 18:
                pattern_score = 1.0
            elif 5 <= transitions < 8 or 18 < transitions <= 25:
                pattern_score = 0.5
            else:
                pattern_score = 0.1

            # 4. Multiplicative Fusion
            safe_density = max(0.1, s_density)
            final_score = (s_geo * 1.5) * (s_hx * s_vy) * safe_density * pattern_score

            candidates.append({
                'box': (x, y, w, h),
                'score': final_score
            })

        if not candidates:
            return None

        best_candidate = sorted(candidates, key=lambda k: k['score'], reverse=True)[0]
        return best_candidate['box']