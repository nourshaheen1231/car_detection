import cv2 as cv
import numpy as np
from collections import deque
import MNN


def resize_and_pad(img, size=(224, 224), pad_color=0):
    h, w = img.shape[:2]
    sh, sw = size

    interp = cv.INTER_AREA if (h > sh or w > sw) else cv.INTER_CUBIC
    aspect = w / h

    if aspect > 1:  # صورة أفقية
        new_w = sw
        new_h = int(np.round(new_w / aspect))
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = int(np.floor(pad_vert)), int(np.ceil(pad_vert))
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # صورة عمودية
        new_h = sh
        new_w = int(np.round(new_h * aspect))
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = int(np.floor(pad_horz)), int(np.ceil(pad_horz))
        pad_top, pad_bot = 0, 0
    else:  # صورة مربعة
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    if len(img.shape) == 3 and not isinstance(pad_color, (list, tuple, np.ndarray)):
        pad_color = [pad_color] * 3

    scaled_img = cv.resize(img, (new_w, new_h), interpolation=interp)
    return cv.copyMakeBorder(
        scaled_img,
        pad_top,
        pad_bot,
        pad_left,
        pad_right,
        borderType=cv.BORDER_CONSTANT,
        value=pad_color,
    )


class CarMakeModelDetection:

    def __init__(
        self,
        model_path: str,
        class_names: list,
        history_size: int = 15,
        input_size: tuple = (128, 128),
        min_confidence: float = 0.4,
        compute_interval: int = 5,
    ):
        self.interpreter = MNN.Interpreter(model_path)
        self.session = self.interpreter.createSession()
        self.input_tensor = self.interpreter.getSessionInput(self.session)

        self.class_names = class_names
        self.input_size = input_size
        self.min_confidence = min_confidence
        self.history_size = history_size
        self.compute_interval = compute_interval

        self._history = {}
        self._cache = {}
        self._frame_counters = {}

        self.last_seen_frame = {}  # لتخزين رقم آخر فريم ظهرت فيه كل سيارة
        self.max_lost_frames = 120  # فترة السماح (مثلاً 120 فريم = 4 ثواني تقريباً)

    # ===============================
    # BATCHED FRAME-LEVEL DETECTION
    # ===============================
    def get_stable_make_models_for_frame(self, frame, car_dict, frame_idx):
        """
        تحديد الشركة والموديل بشكل مجمع لكل الإطار.
        car_dict: {track_id: bbox, ...}
        """
        if not car_dict:
            return {}

        needs_compute = []
        cached_results = {}

        for track_id in car_dict.keys():
            self._frame_counters[track_id] = self._frame_counters.get(track_id, 0) + 1
            self.last_seen_frame[track_id] = frame_idx

            current_count = self._frame_counters[track_id]
            if track_id in self._cache and current_count % self.compute_interval != 0:
                cached = self._cache[track_id]
                cached_results[track_id] = (cached["make_model"], cached["confidence"])
            else:
                needs_compute.append(track_id)

        if not needs_compute:
            return cached_results

        compute_dict = {tid: car_dict[tid] for tid in needs_compute}
        per_car_crops, crop_counts, ordered_ids = self._build_all_crops(frame, compute_dict)

        raw_results = self._run_batched_prediction(
            frame, compute_dict, per_car_crops, crop_counts, ordered_ids
        )

        final_results = self._update_history_and_vote(raw_results)
        final_results.update(cached_results)
        return final_results

    # ===============================
    # BACKWARD COMPATIBILITY
    # ===============================
    def get_stable_make_model(self, track_id, crop, frame_idx):
        """للتوافق مع الاستدعاءات القديمة — تستدعي النسخة المجمعة"""
        if track_id == -1:
            return self._predict_make_model(crop)

        h, w = crop.shape[:2]
        bbox = [0, 0, w, h]
        result = self.get_stable_make_models_for_frame(crop, {track_id: bbox}, frame_idx)
        return result.get(track_id, ("Unknown", 0.0))

    # ===============================
    # BUILD CROPS — Batched
    # ===============================
    def _build_all_crops(self, frame, car_dict):
        """تبني كل الـ crops للسيارات دفعة واحدة."""
        ordered_ids = list(car_dict.keys())
        per_car_crops = []
        crop_counts = []

        for track_id in ordered_ids:
            crops = self._build_crops(frame, car_dict[track_id])
            crop_counts.append(len(crops))
            per_car_crops.extend(crops)

        return per_car_crops, crop_counts, ordered_ids

    def _build_crops(self, frame, bbox):
        """تبني crop واحد لسيارة واحدة (numpy array جاهز للـ MNN)."""
        base = self._crop_roi(frame, bbox)
        if base is None:
            return []

        return [self._preprocess(base)]

    # ===============================
    # BATCHED PREDICTION
    # ===============================
    def _run_batched_prediction(self, frame, car_dict, per_car_crops, crop_counts, ordered_ids):
        """تشغيل النموذج بشكل مجمع على كل الـ crops."""
        raw_results = {}

        if not per_car_crops:
            for track_id in ordered_ids:
                raw_results[track_id] = ("Unknown", 0.0)
            return raw_results

        offset = 0
        for track_id, count in zip(ordered_ids, crop_counts):
            if count == 0:
                raw_results[track_id] = ("Unknown", 0.0)
                continue

            # MNN ما بيدعم الباتشينج المباشر، فمنعالج كل سيارة على حدة
            # لكن الـ preprocessing صار مجمع أولاً (per_car_crops جاهزة)
            car_pred = per_car_crops[offset]
            offset += 1

            tmp_input = MNN.Tensor(
                (1, 3, self.input_size[1], self.input_size[0]),
                MNN.Halide_Type_Float,
                np.ascontiguousarray(car_pred),
                MNN.Tensor_DimensionType_Caffe,
            )

            self.input_tensor.copyFrom(tmp_input)
            self.interpreter.runSession(self.session)

            output_tensor = self.interpreter.getSessionOutput(self.session)
            output_host = MNN.Tensor(output_tensor, MNN.Tensor_DimensionType_Caffe)
            output_tensor.copyToHostTensor(output_host)

            preds = np.array(output_host.getData())
            idx = int(np.argmax(preds))
            conf = float(preds[idx])

            if conf < self.min_confidence or idx >= len(self.class_names):
                raw_results[track_id] = ("Unknown", conf)
            else:
                raw_results[track_id] = (self.class_names[idx], conf)

        return raw_results

    # ===============================
    # TEMPORAL SMOOTHING (Weighted Voting)
    # ===============================
    def _update_history_and_vote(self, raw_results):
        """تحديث التاريخ والتصويت الزمني لكل السيارات."""
        stable_mmrs = {}

        for track_id, (make_model, confidence) in raw_results.items():
            if track_id not in self._history:
                self._history[track_id] = deque(maxlen=self.history_size)

            if make_model != "Unknown":
                self._history[track_id].append((make_model, confidence))

            history = self._history[track_id]

            if not history:
                stable_mmrs[track_id] = ("Unknown", 0.0)
                continue

            # حساب التصويت الموزون بالثقة
            weighted_scores = {}
            for mm, conf in history:
                weighted_scores[mm] = weighted_scores.get(mm, 0.0) + conf

            final_make_model = max(weighted_scores, key=weighted_scores.get)
            total_weight = sum(weighted_scores.values())
            vote_ratio = weighted_scores[final_make_model] / total_weight

            # تطبيق المنطق الذكي الموحد
            if vote_ratio >= 0.5 and final_make_model != "Unknown":
                final_make_model = final_make_model
                final_conf = vote_ratio
            elif make_model != "Unknown" and confidence >= self.min_confidence:
                final_make_model = make_model
                final_conf = confidence
            else:
                final_make_model = "Unknown"
                final_conf = 0.0

            # تحديث الـ Cache بالنتيجة النهائية
            self._cache[track_id] = {
                "make_model": final_make_model,
                "confidence": final_conf,
            }

            stable_mmrs[track_id] = (final_make_model, final_conf)

        return stable_mmrs

    # ===============================
    # SINGLE PREDICTION (Backward compatibility)
    # ===============================
    def _predict_make_model(self, crop):
        if crop is None or crop.size == 0:
            return "Unknown", 0.0

        crop_rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
        crop_padded = resize_and_pad(crop_rgb, self.input_size)

        img_data = crop_padded.astype(np.float32) / 127.5 - 1.0
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)

        tmp_input = MNN.Tensor(
            (1, 3, self.input_size[1], self.input_size[0]),
            MNN.Halide_Type_Float,
            np.ascontiguousarray(img_data),
            MNN.Tensor_DimensionType_Caffe,
        )

        self.input_tensor.copyFrom(tmp_input)
        self.interpreter.runSession(self.session)

        output_tensor = self.interpreter.getSessionOutput(self.session)
        output_host = MNN.Tensor(output_tensor, MNN.Tensor_DimensionType_Caffe)
        output_tensor.copyToHostTensor(output_host)

        preds = np.array(output_host.getData())

        idx = int(np.argmax(preds))
        conf = float(preds[idx])

        if conf < self.min_confidence or idx >= len(self.class_names):
            return "Unknown", conf

        return self.class_names[idx], conf

    # ===============================
    # CROP + PREPROCESS
    # ===============================
    def _crop_roi(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(x1, 0), max(y1, 0)

        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _preprocess(self, crop_bgr):
        crop_rgb = cv.cvtColor(crop_bgr, cv.COLOR_BGR2RGB)
        crop_padded = resize_and_pad(crop_rgb, self.input_size)

        img_data = crop_padded.astype(np.float32) / 127.5 - 1.0
        img_data = np.transpose(img_data, (2, 0, 1))
        return np.expand_dims(img_data, axis=0)

    # ===============================
    # CLEANUP
    # ===============================
    def cleanup_inactive_tracks(self, current_frame_idx):
        """
        يحذف بيانات السيارات التي غابت عن الشاشة لفترة تتجاوز max_lost_frames
        """
        lost_ids = []

        for tid, last_frame in list(self.last_seen_frame.items()):
            if (current_frame_idx - last_frame) > self.max_lost_frames:
                lost_ids.append(tid)

        for tid in lost_ids:
            if tid in self._cache:
                del self._cache[tid]
            if tid in self._frame_counters:
                del self._frame_counters[tid]
            if tid in self._history:
                del self._history[tid]
            del self.last_seen_frame[tid]

    def reset(self):
        """تفريغ كلي لجميع القواميس"""
        self._history.clear()
        self._cache.clear()
        self._frame_counters.clear()
        self.last_seen_frame.clear()