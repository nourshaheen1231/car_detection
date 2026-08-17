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
        min_plate_confidence=0.5,
        enable_ocr=True,
        ocr_retry_interval=5,
        plate_padding=3,
        max_gap_frames=2,
        jump_iou_threshold=0.3,
    ):

        self.max_lost_frames = max_lost_frames
        self.plate_roi_ratio = plate_roi_ratio
        self.min_plate_confidence = min_plate_confidence
        self.enable_ocr = enable_ocr
        self.ocr_retry_interval = ocr_retry_interval
        self.plate_padding = plate_padding
        self.max_gap_frames = max_gap_frames
        self.jump_iou_threshold = jump_iou_threshold

        self.plate_tracks = {}
        self.car_to_plate = {}
        self.next_plate_id = 101

        self.last_frame_stats = {
            "cv_detect_calls": 0,
            "cv_cars_inferred": 0,
            "confirmed_car_ids": [],
            "buffered_car_ids": [],
            "pending_jump_car_ids": [],
            "cleared_car_ids": [],
        }

    def _new_plate_track(self, vehicle_id):
        return {
            "vehicle_id": vehicle_id,
            "ocr_done": False,
            "ocr_attempts": 0,
            "last_ocr_frame": -10**9,
            "text": None,
            "text_conf": None,
            "last_confirmed_bbox": None,
            "confidence": None,
            "miss_streak": 0,
            "pending_bbox": None,
            "last_seen": -10**9,
            "rel_bbox": None,
        }


    def relative_bbox(self, vehicle_bbox, plate_bbox):
        xcar1, ycar1, xcar2, ycar2 = vehicle_bbox
        car_width = max(xcar2 - xcar1, 1)
        car_height = max(ycar2 - ycar1, 1)
        x1, y1, x2, y2 = plate_bbox
        return [(x1 - xcar1) / car_width, (y1 - ycar1) / car_height,
                (x2 - xcar1) / car_width, (y2 - ycar1) / car_height]

    def project_bbox(self, vehicle_bbox, rel_bbox):
        xcar1, ycar1, xcar2, ycar2 = vehicle_bbox
        car_width = xcar2 - xcar1
        car_height = ycar2 - ycar1
        rx1, ry1, rx2, ry2 = rel_bbox
        return [xcar1 + rx1 * car_width, ycar1 + ry1 * car_height,
                xcar1 + rx2 * car_width, ycar1 + ry2 * car_height]

    def plate_is_lost(self, plate_track, plate_bbox, frame_num, frame_size):
       
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

    # BATCHED DETECTION

    def detect_plates_for_frame(self, car_rois):

        results = []
        for car_roi in car_rois:
            plate_box = self.detect_plate_location_dynamic_guassanian(car_roi)
            if plate_box is None:
                results.append(None)
            else:
                x, y, w, h = plate_box
                results.append((x, y, w, h, 0.5))
        return results


    @staticmethod
    def _iou(box_a, box_b):
        if box_a is None or box_b is None:
            return 0.0

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0

        return inter_area / union

    def _update_plate_track_state(self, plate_track, fresh_bbox, fresh_conf):

        last_confirmed = plate_track.get('last_confirmed_bbox')

        if fresh_bbox is None:
            plate_track['pending_bbox'] = None
            plate_track['miss_streak'] = plate_track.get('miss_streak', 0) + 1
            if last_confirmed is not None and plate_track['miss_streak'] <= self.max_gap_frames:
                return last_confirmed, 'buffered'
            return None, 'cleared'

        if last_confirmed is None:
            plate_track['last_confirmed_bbox'] = fresh_bbox
            plate_track['confidence'] = fresh_conf
            plate_track['miss_streak'] = 0
            plate_track['pending_bbox'] = None
            return fresh_bbox, 'confirmed'

        if self._iou(fresh_bbox, last_confirmed) >= self.jump_iou_threshold:
            plate_track['last_confirmed_bbox'] = fresh_bbox
            plate_track['confidence'] = fresh_conf
            plate_track['miss_streak'] = 0
            plate_track['pending_bbox'] = None
            return fresh_bbox, 'confirmed'

        pending = plate_track.get('pending_bbox')
        if pending is not None and self._iou(fresh_bbox, pending) >= self.jump_iou_threshold:
            plate_track['last_confirmed_bbox'] = fresh_bbox
            plate_track['confidence'] = fresh_conf
            plate_track['miss_streak'] = 0
            plate_track['pending_bbox'] = None
            return fresh_bbox, 'confirmed'

        plate_track['pending_bbox'] = fresh_bbox
        plate_track['miss_streak'] = plate_track.get('miss_streak', 0) + 1
        if plate_track['miss_streak'] <= self.max_gap_frames:
            return last_confirmed, 'pending_jump'
        return None, 'cleared'

    def _freeze_relative_position(self, plate_track, car_bbox):

        if plate_track.get('rel_bbox') is not None:
            return  # اتجمّد من قبل، ما نلمسه تاني
        anchor_bbox = plate_track.get('last_confirmed_bbox')
        if anchor_bbox is None:
            return
        plate_track['rel_bbox'] = self.relative_bbox(car_bbox, anchor_bbox)

    def track_plates_for_frame(self, frame, car_dict, frame_idx, frame_size):

        self.last_frame_stats = {
            "cv_detect_calls": 0,
            "cv_cars_inferred": 0,
            "confirmed_car_ids": [],
            "buffered_car_ids": [],
            "pending_jump_car_ids": [],
            "cleared_car_ids": [],
        }

        results = {}

        plate_rois = []
        detect_order = []

        for track_id, car_info in car_dict.items():
            plate_id = self.car_to_plate.get(track_id)
            if plate_id is not None:
                plate_track = self.plate_tracks.get(plate_id)
                if plate_track is not None and plate_track.get('ocr_done', False) and plate_track.get('last_confirmed_bbox') is not None:
                    continue

            crop = car_info['crop']
            plate_roi, y_offset = self._crop_plate_roi(crop)
            if plate_roi is not None:
                plate_rois.append(plate_roi)
                detect_order.append((track_id, car_info, y_offset))

        detections = []
        if plate_rois:
            self.last_frame_stats["cv_detect_calls"] = 1
            self.last_frame_stats["cv_cars_inferred"] = len(plate_rois)
            detections = self.detect_plates_for_frame(plate_rois)

        fresh_by_track = {}
        for (track_id, car_info, y_offset), plate_local_box in zip(detect_order, detections):
            if plate_local_box is None:
                fresh_by_track[track_id] = (None, None)
                continue

            px, py, pw, ph, plate_conf = plate_local_box
            x1_car, y1_car, _, _ = self._clip_bbox(frame, car_info['bbox'])
            plate_bbox = [
                x1_car + px,
                y1_car + y_offset + py,
                x1_car + px + pw,
                y1_car + y_offset + py + ph,
            ]
            fresh_by_track[track_id] = (plate_bbox, plate_conf)

        for track_id in car_dict:
            if track_id not in self.car_to_plate:
                plate_id = self.next_plate_id
                self.next_plate_id += 1
                self.car_to_plate[track_id] = plate_id
                self.plate_tracks[plate_id] = self._new_plate_track(track_id)

            plate_id = self.car_to_plate[track_id]
            plate_track = self.plate_tracks[plate_id]

            plate_track['last_seen'] = frame_idx

            if plate_track.get('ocr_done', False) and plate_track.get('last_confirmed_bbox') is not None:
                if plate_track.get('rel_bbox') is not None:
                    display_bbox = self.project_bbox(car_dict[track_id]['bbox'], plate_track['rel_bbox'])
                else:
                    display_bbox = plate_track['last_confirmed_bbox']
                results[track_id] = display_bbox
                self.last_frame_stats["confirmed_car_ids"].append(track_id)
                continue

            fresh_bbox, fresh_conf = fresh_by_track.get(track_id, (None, None))
            bbox_to_report, state = self._update_plate_track_state(plate_track, fresh_bbox, fresh_conf)

            if bbox_to_report is not None:
                results[track_id] = bbox_to_report

            self.last_frame_stats[f"{state}_car_ids"].append(track_id)

        lost_cars = []
        for track_id, plate_id in list(self.car_to_plate.items()):
            if track_id not in car_dict:
                if plate_id in self.plate_tracks:
                    if frame_idx - self.plate_tracks[plate_id]['last_seen'] > self.max_lost_frames:
                        lost_cars.append(track_id)

        for track_id in lost_cars:
            plate_id = self.car_to_plate.pop(track_id, None)
            if plate_id in self.plate_tracks:
                del self.plate_tracks[plate_id]

        self._run_ocr_for_active_plates(frame, frame_idx, frame_size, results)

        for track_id, car_info in car_dict.items():
            plate_id = self.car_to_plate.get(track_id)
            if plate_id and self.plate_tracks[plate_id].get('ocr_done'):
                self._freeze_relative_position(self.plate_tracks[plate_id], car_info['bbox'])

        return results

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
        self.plate_tracks.clear()
        self.car_to_plate.clear()
        self.next_plate_id = 101

    def _crop_plate_roi(self, car_crop):
        if car_crop is None or car_crop.size == 0:
            return None, 0
        h, w = car_crop.shape[:2]
        y_offset = int(h * self.plate_roi_ratio)
        return car_crop[y_offset:h, 0:w], y_offset

    def _clip_bbox(self, frame, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return x1, y1, x2, y2

    def _calculate_projection_profile_transitions(self, edge_roi: np.ndarray, w: int, h: int) -> int:

        strip_y1, strip_y2 = int(h * 0.33), int(h * 0.66)
        if strip_y2 <= strip_y1:
            return 0 
        mid_strip = edge_roi[strip_y1:strip_y2, :]
        strip_h = strip_y2 - strip_y1

        col_sums = np.sum(mid_strip > 0, axis=0)

        col_binary = (col_sums > (strip_h * 0.2)).astype(np.uint8)

        transitions = np.sum(col_binary[:-1] != col_binary[1:])
        return transitions

    def detect_plate_location_dynamic_guassanian(self, car_roi: np.ndarray) -> tuple | None:

        if car_roi is None or car_roi.size == 0:
            return None

        h_car, w_car = car_roi.shape[:2]
        car_area = w_car * h_car
        roi_center_x = w_car / 2.0

        kernel_w = max(3, int(w_car * 0.035))
        kernel_h = max(2, int(h_car * 0.02))

        min_plate_area = car_area * 0.005
        max_plate_area = car_area * 0.15

        gray = cv.cvtColor(car_roi, cv.COLOR_BGR2GRAY)

        # Adaptive Thresholding 
        blurred_a = cv.GaussianBlur(gray, (3, 3), 0)
        char_binary = cv.adaptiveThreshold(
            blurred_a, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv.THRESH_BINARY_INV, 15, 7
        )

        kernel_merge = cv.getStructuringElement(cv.MORPH_RECT, (kernel_w + 5, 2))
        merged_strip = cv.morphologyEx(char_binary, cv.MORPH_CLOSE, kernel_merge)

        kernel_clean = cv.getStructuringElement(cv.MORPH_RECT, (1, max(3, kernel_h // 2)))
        closed_contours_mask = cv.morphologyEx(merged_strip, cv.MORPH_OPEN, kernel_clean)

        #  Texture Extraction 
        gray_smooth_b = cv.GaussianBlur(gray, (5, 5), 0)
        tophat_kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 5))
        tophat = cv.morphologyEx(gray_smooth_b, cv.MORPH_TOPHAT, tophat_kernel)

        blurred_b = cv.GaussianBlur(tophat, (5, 5), 0)
        sobel_x = cv.Sobel(blurred_b, cv.CV_16S, dx=1, dy=0, ksize=3)
        abs_sobel_x = cv.convertScaleAbs(sobel_x)

        _, threshed = cv.threshold(abs_sobel_x, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Feature Extraction
        contours, _ = cv.findContours(closed_contours_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        candidates = []

        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            box_area = w * h

            # 1. Hard Geometry Gates 
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

            # 2. Hard Density Gates 
            edge_roi = threshed[y:y+h, x:x+w]
            white_pixels = cv.countNonZero(edge_roi)
            density_ratio = white_pixels / box_area

            if white_pixels < 15 or density_ratio < 0.08:
                continue 

            # Complex Scoring Functions 

            # (Shape)
            ar_score = np.exp(-((aspect_ratio - 3.5) ** 2) / 2.0)
            s_geo = ar_score * extent  

            # (Spatial X & Y)
            box_center_x = x + (w / 2.0)
            distance_from_center_x = abs(roi_center_x - box_center_x)
            s_hx = max(0, 1.0 - (distance_from_center_x / roi_center_x)) 

            box_center_y = y + (h / 2.0)
            y_norm = box_center_y / h_car 
            s_vy = np.exp(-((y_norm - 0.65) ** 2) / 0.15) # دالة ذروة واسعة النطاق

            # (Texture Density)
            s_density = np.exp(-((density_ratio - 0.30) ** 2) / 0.04) # دالة ذروة تستهدف 30%

            # (Text Pattern Consistency)
            transitions = self._calculate_projection_profile_transitions(edge_roi, w, h)

            if 8 <= transitions <= 20:
                pattern_score = 1.0
            elif 5 <= transitions < 8 or 20 < transitions <= 28:
                pattern_score = 0.5
            else:
                pattern_score = 0.01 

            #(Multiplicative Fusion)
            final_score = (s_geo * 1.5) * (s_hx * s_vy) * s_density * pattern_score

            candidates.append({
                'box': (x, y, w, h),
                'score': final_score
            })

        if not candidates:
            return None

        best_candidate = sorted(candidates, key=lambda k: k['score'], reverse=True)[0]
        return best_candidate['box']