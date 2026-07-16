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
                 min_confidence: float = 0.4,
                 compute_interval: int = 5):   

        self.model = load_model(model_path)
        self.input_size = input_size
        self.rescale = rescale
        self.min_confidence = min_confidence
        self.history_size = history_size
        self.compute_interval = compute_interval

        self._color_history = {}
        self._color_cache = {}    
        self._frame_counter = 0

    # ===============================
    # MAIN FUNCTION
    # ===============================
    def get_stable_color(self, track_id, frame, bbox):

        self._frame_counter += 1

        if track_id in self._color_cache:
            if self._frame_counter % self.compute_interval != 0:
                return self._color_cache[track_id]["color"]

        color, confidence = self._predict_color(frame, bbox)

        if color != "Unknown":
            self._color_cache[track_id] = {
                "color": color,
                "confidence": confidence
            }

        print(f"[Car {track_id}] Color: {color} | Confidence: {confidence:.2f}")

        # ===============================
        # Temporal smoothing
        # ===============================
        if track_id not in self._color_history:
            self._color_history[track_id] = deque(maxlen=self.history_size)

        if color != "Unknown":
            self._color_history[track_id].append(color)

        if not self._color_history[track_id]:
            return "Unknown"

        final_color = Counter(self._color_history[track_id]).most_common(1)[0][0]

        self._color_cache[track_id]["color"] = final_color

        return final_color

    # ===============================
    # COLOR PREDICTION
    # ===============================
    def _predict_color(self, frame, bbox):

        crop = self._crop(frame, bbox)
        if crop is None:
            return "Unknown", 0.0

        # ===============================
        # CNN
        # ===============================
        cnn_color, cnn_conf = self._predict_cnn(crop)

        # ===============================
        # KMeans 
        # ===============================
        if cnn_conf < 0.6:
            kmeans_color = self._predict_kmeans(crop)
            return kmeans_color, cnn_conf

        return cnn_color, cnn_conf

    # ===============================
    # CNN
    # ===============================
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

    # ===============================
    # KMeans
    # ===============================
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

    # ===============================
    # COLOR MAP
    # ===============================
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
    # CROP
    # ===============================
    def _crop(self, frame, bbox):

        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(x1, 0), max(y1, 0)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return crop

    def reset(self):
        self._color_history.clear()
        self._color_cache.clear()
        self._frame_counter = 0