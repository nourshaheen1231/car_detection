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
        self._frame_counters = {}
    # ===============================
    # MAIN FUNCTION
    # ===============================
    def get_stable_color(self, track_id, frame, bbox):

        # زيادة العداد الخاص بهذه السيارة تحديداً
        self._frame_counters[track_id] = self._frame_counters.get(track_id, 0) + 1
        current_count = self._frame_counters[track_id]

        # 1. التحقق من الـ Cache (لتخفيف الضغط وعدم الحساب في كل إطار)
        if track_id in self._color_cache:
            if current_count % self.compute_interval != 0:
                return self._color_cache[track_id]["color"], self._color_cache[track_id].get("confidence", 0.0)

        # 2. التنبؤ باللون الجديد (استخدمنا اسم raw_color لتجنب التضارب)
        raw_color, raw_confidence = self._predict_color(frame, bbox)

        print(f"[Car {track_id}] Color: {raw_color} | Confidence: {raw_confidence:.2f}")

        # ===============================
        # Temporal smoothing (التصويت الزمني)
        # ===============================
        
        # الطريقة الأولى: إنشاء القائمة إن لم تكن موجودة لهذا الـ track_id
        if track_id not in self._color_history:
            self._color_history[track_id] = deque(maxlen=self.history_size)

        # إضافة النتيجة إلى السجل فقط إذا لم تكن غير معروفة
        if raw_color != "Unknown":
            self._color_history[track_id].append((raw_color, raw_confidence))

        # إذا كان السجل ما زال فارغاً (مثلاً أول إطار وكان التنبؤ Unknown)
        if not self._color_history[track_id]:
            return "Unknown", 0.0

        # حساب التصويت بناءً على السجل
        weighted_scores = {}
        buffer = self._color_history[track_id]
        
        for c, conf in buffer:
            weighted_scores[c] = weighted_scores.get(c, 0.0) + conf
            
        final_color = max(weighted_scores, key=weighted_scores.get)
        total_weight = sum(weighted_scores.values())
        vote_ratio = weighted_scores[final_color] / total_weight

        # 3. تحديث الـ Cache بالنتيجة النهائية (المستقرة)
        self._color_cache[track_id] = {
            "color": final_color,
            "confidence": vote_ratio
        }

        return final_color, vote_ratio

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



    def cleanup_inactive_tracks(self, active_track_ids):
        """
        تستقبل قائمة الـ track_ids التابعة للسيارات الموجودة حالياً في الفريم،
        وتحذف بيانات أي سيارة اختفت لمنع استهلاك الذاكرة.
        """
        active_set = set(active_track_ids)

        # 1. تنظيف كاش اللون
        for tid in list(self._color_cache.keys()):
            if tid not in active_set:
                del self._color_cache[tid]

        # 2. تنظيف عداد الفريمات
        for tid in list(self._frame_counters.keys()):
            if tid not in active_set:
                del self._frame_counters[tid]

        # 3. تنظيف سجل التاريخ الزمني (History)
        for tid in list(self._color_history.keys()):
            if tid not in active_set:
                del self._color_history[tid]

    def reset(self):
        """تفريغ كلي لجميع القواميس مع تصحيح اسم عداد الفريمات"""
        self._color_history.clear()
        self._color_cache.clear()
        self._frame_counters.clear()  # تم تصحيح المفرد إلى الجمع _frame_counters