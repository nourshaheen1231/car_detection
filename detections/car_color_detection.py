import cv2 as cv
import numpy as np
from collections import deque, Counter
from keras.models import load_model
from sklearn.cluster import KMeans


class CarColorDetection:

    CLASS_NAMES = [
        "beige", "black", "blue", "brown", "gold", "green", "grey",
        "orange", "pink", "purple", "red", "silver", "tan", "white",
        "yellow",
    ]

    def __init__(self,
                 model_path: str,
                 history_size: int = 15,
                 input_size: tuple = (128, 128),
                 rescale: float = 1.0 / 255.0,
                 min_confidence: float = 0.55,
                 compute_interval: int = 5,
                 min_crop_size: int = 25,
                 use_tta: bool = True,
                 roi_margin: float = 0.0):

        self.model = load_model(model_path)
        self.input_size = input_size
        self.rescale = rescale
        self.min_confidence = min_confidence
        self.compute_interval = compute_interval
        self.history_size = history_size
        self.min_crop_size = min_crop_size
        self.use_tta = use_tta
        self.roi_margin = roi_margin

        self._color_history = {}
        self._color_cache = {}
        self._frame_counters = {}
        self._kmeans_cache = {}

        self.last_seen_frame = {}  # لتخزين رقم آخر فريم ظهرت فيه كل سيارة
        self.max_lost_frames = 120  # فترة السماح (مثلاً 120 فريم = 4 ثواني تقريباً)

        # عتبات KMeans fallback
        self.kmeans_conf_threshold = 0.5
        self.kmeans_sample_size = 500

    # ===============================
    # MAIN FUNCTION — Frame-level Batching
    # ===============================
    def get_stable_colors_for_frame(self, frame, car_dict: dict, frame_idx: int):
        """
        تعالج كل السيارات الموجودة بالإطار دفعة واحدة (Batched).
        car_dict: {track_id: bbox, ...}
        """
        if not car_dict:
            return {}

        # تحديث العدادات وآخر فريم مشاهدة
        needs_compute = []
        cached_results = {}

        for track_id in car_dict.keys():
            self._frame_counters[track_id] = self._frame_counters.get(track_id, 0) + 1
            self.last_seen_frame[track_id] = frame_idx

            current_count = self._frame_counters[track_id]
            # إذا الكاش موجود والفاصل الزمني ما وصل، رجع الكاش مباشرة
            if track_id in self._color_cache and current_count % self.compute_interval != 0:
                cached = self._color_cache[track_id]
                cached_results[track_id] = (cached["color"], cached.get("confidence", 0.0))
            else:
                needs_compute.append(track_id)

        # إذا كل السيارات عندن كاش صالح، رجع الكاش
        if not needs_compute:
            return cached_results

        # بني الـ crops للسيارات يلي بحاجة حساب
        compute_dict = {tid: car_dict[tid] for tid in needs_compute}
        per_car_crops, crop_counts, ordered_ids = self._build_all_crops(frame, compute_dict)

        # تشغيل النموذج بشكل مجمع (Batched)
        raw_results = self._run_batched_prediction(
            frame, compute_dict, per_car_crops, crop_counts, ordered_ids
        )

        # التصويت الزمني وتحديث الكاش
        final_results = self._update_history_and_vote(raw_results)

        # دمج النتائج الجديدة مع الكاش القديم
        final_results.update(cached_results)
        return final_results

    # ===============================
    # BACKWARD COMPATIBILITY
    # ===============================
    def get_stable_color(self, track_id, crop, frame_idx):
        """للتوافق مع الاستدعاءات القديمة — تستدعي النسخة المجمعة"""
        if track_id == -1:
            return self._predict_color(crop)

        # نبني bbox وهمي من الـ crop (مش مثالي بس للتوافق)
        h, w = crop.shape[:2]
        bbox = [0, 0, w, h]
        result = self.get_stable_colors_for_frame(crop, {track_id: bbox}, frame_idx)
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
        """تبني crop واحد أو أكثر (مع TTA) لسيارة واحدة."""
        base = self._crop_roi(frame, bbox)
        if base is None:
            return []

        crops = [self._preprocess(base)]

        if self.use_tta:
            flipped = cv.flip(base, 1)
            crops.append(self._preprocess(flipped))

        return crops

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

        batch = np.concatenate(per_car_crops, axis=0)
        predictions = self._run_model(batch)

        offset = 0
        for track_id, count in zip(ordered_ids, crop_counts):
            if count == 0:
                raw_results[track_id] = ("Unknown", 0.0)
                continue

            car_preds = predictions[offset:offset + count]
            offset += count

            avg_pred = car_preds.mean(axis=0)

            best_idx = int(np.argmax(avg_pred))
            confidence = float(avg_pred[best_idx])
            color_name = self.CLASS_NAMES[best_idx]

            # =========================================
            # KMeans fallback (من تعديل زميل اللون)
            # =========================================
            if confidence < self.kmeans_conf_threshold:
                if track_id in self._kmeans_cache:
                    k_color, k_conf = self._kmeans_cache[track_id]
                else:
                    crop = self._crop_roi(frame, car_dict[track_id])
                    if crop is not None:
                        k_color, k_conf = self._predict_kmeans_fast(crop)
                        self._kmeans_cache[track_id] = (k_color, k_conf)
                    else:
                        k_color, k_conf = "Unknown", 0.0

                if k_conf > confidence:
                    color_name = k_color
                    confidence = k_conf

            if confidence < self.min_confidence:
                raw_results[track_id] = ("Unknown", 0.0)
            else:
                raw_results[track_id] = (color_name, confidence)

        return raw_results

    def _run_model(self, batch):
        """تشغيل النموذج على batch من الصور."""
        output = self.model(batch, training=False)
        return output.numpy() if hasattr(output, "numpy") else np.asarray(output)

    # ===============================
    # TEMPORAL SMOOTHING (Weighted Voting)
    # ===============================
    def _update_history_and_vote(self, raw_results):
        """تحديث التاريخ والتصويت الزمني لكل السيارات."""
        stable_colors = {}

        for track_id, (color, conf) in raw_results.items():
            if track_id not in self._color_history:
                self._color_history[track_id] = deque(maxlen=self.history_size)

            if color != "Unknown":
                self._color_history[track_id].append((color, conf))

            history = self._color_history[track_id]

            if not history:
                stable_colors[track_id] = ("Unknown", 0.0)
                continue

            # حساب التصويت بناءً على السجل
            weighted_scores = {}
            for c, confidence in history:
                weighted_scores[c] = weighted_scores.get(c, 0.0) + confidence

            final_color = max(weighted_scores, key=weighted_scores.get)
            total_weight = sum(weighted_scores.values())
            vote_ratio = weighted_scores[final_color] / total_weight

            # تطبيق المنطق الذكي الموحد
            if vote_ratio >= 0.5 and final_color != "Unknown":
                final_color = final_color
                final_conf = vote_ratio
            elif color != "Unknown" and conf >= self.min_confidence:
                final_color = color
                final_conf = conf
            else:
                final_color = "Unknown"
                final_conf = 0.0

            # تحديث الـ Cache بالنتيجة النهائية
            self._color_cache[track_id] = {
                "color": final_color,
                "confidence": final_conf
            }

            stable_colors[track_id] = (final_color, final_conf)

        return stable_colors

    # ===============================
    # FAST KMeans (من تعديل زميل اللون)
    # ===============================
    def _predict_kmeans_fast(self, crop):
        lab = cv.cvtColor(crop, cv.COLOR_BGR2LAB)
        pixels = lab.reshape((-1, 3))

        if len(pixels) > self.kmeans_sample_size:
            idx = np.random.choice(len(pixels), self.kmeans_sample_size, replace=False)
            pixels = pixels[idx]

        kmeans = KMeans(n_clusters=3, n_init=2)
        labels = kmeans.fit_predict(pixels)

        counts = np.bincount(labels)
        dominant_idx = np.argmax(counts)
        dominant = kmeans.cluster_centers_[dominant_idx]
        confidence = counts[dominant_idx] / np.sum(counts)

        color = self._lab_to_color_name(dominant)
        return color, confidence

    # ===============================
    # COLOR PREDICTION (Backward compatibility)
    # ===============================
    def _predict_color(self, crop):
        if crop is None or crop.size == 0:
            return "Unknown", 0.0

        cnn_color, cnn_conf = self._predict_cnn(crop)

        if cnn_conf < 0.6:
            kmeans_color = self._predict_kmeans(crop)
            return kmeans_color, cnn_conf

        return cnn_color, cnn_conf

    def _predict_cnn(self, crop):
        crop_rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
        crop_resized = cv.resize(crop_rgb, self.input_size)

        arr = crop_resized.astype(np.float32) * self.rescale
        arr = np.expand_dims(arr, axis=0)

        preds = self.model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx])

        if conf < self.min_confidence:
            return "Unknown", conf

        return self.CLASS_NAMES[idx], conf

    def _predict_kmeans(self, crop):
        h = crop.shape[0]
        crop = crop[int(h * 0.25):int(h * 0.75), :]

        lab = cv.cvtColor(crop, cv.COLOR_BGR2LAB)
        pixels = lab.reshape((-1, 3))

        kmeans = KMeans(n_clusters=3, n_init=3)
        labels = kmeans.fit_predict(pixels)

        counts = np.bincount(labels)
        dominant = kmeans.cluster_centers_[np.argmax(counts)]

        return self._lab_to_color_name(dominant)

    def _lab_to_color_name(self, lab):
        l, a, b = lab

        if l > 200:
            return "white"
        if l < 50:
            return "black"

        if a > 150:
            return "red"
        if b > 150:
            return "yellow"

        return "grey"

    # ===============================
    # CROP + PREPROCESS
    # ===============================
    def _crop_roi(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(x1, 0), max(y1, 0)

        width, height = x2 - x1, y2 - y1

        if width < self.min_crop_size or height < self.min_crop_size:
            return None

        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _preprocess(self, crop_bgr):
        crop_rgb = cv.cvtColor(crop_bgr, cv.COLOR_BGR2RGB)
        crop_resized = cv.resize(crop_rgb, self.input_size)

        arr = crop_resized.astype(np.float32) * self.rescale
        return np.expand_dims(arr, axis=0)

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
            if tid in self._color_cache:
                del self._color_cache[tid]
            if tid in self._frame_counters:
                del self._frame_counters[tid]
            if tid in self._color_history:
                del self._color_history[tid]
            if tid in self._kmeans_cache:
                del self._kmeans_cache[tid]
            del self.last_seen_frame[tid]

    def reset(self):
        """تفريغ كلي لجميع القواميس"""
        self._color_history.clear()
        self._color_cache.clear()
        self._frame_counters.clear()
        self._kmeans_cache.clear()
        self.last_seen_frame.clear()