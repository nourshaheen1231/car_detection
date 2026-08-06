import cv2 as cv
import numpy as np
from collections import deque
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
                 min_confidence: float = 0.40,
                 min_crop_size: int = 25,
                 use_tta: bool = True,
                 roi_margin: float = 0.0,
                 profiler=None):   


        self.model = load_model(model_path)
        self.input_size = input_size
        self.rescale = rescale
        self.min_confidence = min_confidence
        self.min_crop_size = min_crop_size
        self.use_tta = use_tta
        self.roi_margin = roi_margin
        self.history_size = history_size

        self._color_history = {}
        self._kmeans_cache = {}

        self.kmeans_conf_threshold = 0.5
        self.kmeans_sample_size = 500

        
        self.model = load_model(model_path)
        self.profiler = profiler  


    def get_stable_colors_for_frame(self, frame, car_dict: dict):

        if not car_dict:
            return {}

        if self.profiler:
            with self.profiler.timer("color_crop_tta_build"):
                per_car_crops, crop_counts, ordered_ids = self._build_all_crops(frame, car_dict)
        else:
            per_car_crops, crop_counts, ordered_ids = self._build_all_crops(frame, car_dict)

        raw_results = self._run_batched_prediction(
            frame, car_dict, per_car_crops, crop_counts, ordered_ids
        )

        return self._update_history_and_vote(raw_results)

    def get_stable_color(self, track_id, frame, bbox):
        result = self.get_stable_colors_for_frame(frame, {track_id: bbox})
        return result.get(track_id, "Unknown")

    def reset(self):
        self._color_history.clear()
        self._kmeans_cache.clear()

    # =========================================================
    # BUILD CROPS
    # =========================================================
    def _build_all_crops(self, frame, car_dict):

        ordered_ids = list(car_dict.keys())
        per_car_crops = []
        crop_counts = []
        

        for track_id in ordered_ids:
            crops = self._build_crops(frame, car_dict[track_id])
            crop_counts.append(len(crops))
            per_car_crops.extend(crops)

        return per_car_crops, crop_counts, ordered_ids

    def _build_crops(self, frame, bbox):

        base = self._crop_roi(frame, bbox)
        if base is None:
            return []

        crops = [self._preprocess(base)]

        if self.use_tta:
            flipped = cv.flip(base, 1)
            crops.append(self._preprocess(flipped))

        return crops

    def _run_batched_prediction(self, frame, car_dict, per_car_crops, crop_counts, ordered_ids):

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
            #  KMeans fallback 
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

        if self.profiler:
            with self.profiler.timer("color_model_predict"):
                output = self.model(batch, training=False)
        else:
            output = self.model(batch, training=False)

        return output.numpy() if hasattr(output, "numpy") else np.asarray(output)


    # =========================================================
    # TEMPORAL SMOOTHING (Weighted Voting)
    # =========================================================
    def _update_history_and_vote(self, raw_results):

        stable_colors = {}

        for track_id, (color, conf) in raw_results.items():

            if track_id not in self._color_history:
                self._color_history[track_id] = deque(maxlen=self.history_size)

            if color != "Unknown":
                self._color_history[track_id].append((color, conf))

            history = self._color_history[track_id]

            if not history:
                stable_colors[track_id] = "Unknown"
                continue

            scores = {}
            for c, confidence in history:
                scores[c] = scores.get(c, 0) + confidence

            if self.profiler:
                with self.profiler.timer("color_voting"):
                    stable_colors[track_id] = max(scores, key=scores.get)
            else:
                stable_colors[track_id] = max(scores, key=scores.get)

        return stable_colors
    # =========================================================
    # FAST KMeans 
    # =========================================================


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

    # =========================================================
    # CROP + PREPROCESS
    # =========================================================
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