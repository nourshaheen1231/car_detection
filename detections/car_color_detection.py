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

        self.last_seen_frame = {}  # لتخزين رقم آخر فريم ظهرت فيه كل سيارة
        self.max_lost_frames = 120  # فترة السماح (مثلاً 120 فريم = 4 ثواني تقريباً)
    # ===============================
    # MAIN FUNCTION
    # ===============================
    def get_stable_color(self, track_id, crop, frame_idx):
        if track_id == -1:
            return self._predict_color(crop)
        # زيادة العداد الخاص بهذه السيارة تحديداً
        self._frame_counters[track_id] = self._frame_counters.get(track_id, 0) + 1
        current_count = self._frame_counters[track_id]

        # [جديد] تسجيل آخر فريم ظهرت فيه هذه السيارة
        self.last_seen_frame[track_id] = frame_idx

        # 1. التحقق من الـ Cache (لتخفيف الضغط وعدم الحساب في كل إطار)
        if track_id in self._color_cache:
            if current_count % self.compute_interval != 0:
                return self._color_cache[track_id]["color"], self._color_cache[track_id].get("confidence", 0.0)

        # 2. التنبؤ باللون الجديد (استخدمنا اسم raw_color لتجنب التضارب)
        raw_color, raw_confidence = self._predict_color(crop)

        # print(f"[Car {track_id}] Color: {raw_color} | Confidence: {raw_confidence:.2f}")

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

        # 5. تطبيق المنطق الذكي الموحد
        if vote_ratio >= 0.5 and final_color != "Unknown":
            # الحالة الأولى: التصويت الزمني حاسم والأغلبية متفقة
            final_color = final_color
            final_conf = vote_ratio

        elif raw_color != "Unknown" and raw_confidence >= self.min_confidence:
            # الحالة الثانية: التصويت لم يحسم بعد، لكن الفريم الحالي واضح جداً (ثقة أعلى من 70%)
            final_color = raw_color
            final_conf = raw_confidence

        else:
            # الحالة الثالثة: التصويت تائه والفريم الحالي ضعيف، نستسلم
            final_color = "Unknown"
            final_conf = 0.0

        # 6. تحديث الـ Cache بالنتيجة النهائية
        self._color_cache[track_id] = {
            "color": final_color,
            "confidence": final_conf
        }

        return final_color, final_conf

    # ===============================
    # COLOR PREDICTION
    # ===============================
    def _predict_color(self, crop):

        if crop is None or crop.size == 0:
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
    # def _crop(self, frame, bbox):

    #     x1, y1, x2, y2 = map(int, bbox)
    #     x1, y1 = max(x1, 0), max(y1, 0)

    #     crop = frame[y1:y2, x1:x2]

    #     if crop.size == 0:
    #         return None

    #     return crop



    def cleanup_inactive_tracks(self, current_frame_idx):
        """
        يحذف بيانات السيارات التي غابت عن الشاشة لفترة تتجاوز max_lost_frames
        """
        lost_ids = []
        
        # تحديد السيارات المفقودة بناءً على آخر فريم شوهدت فيه
        for tid, last_frame in list(self.last_seen_frame.items()):
            if (current_frame_idx - last_frame) > self.max_lost_frames:
                lost_ids.append(tid)

        # تنظيف شامل للسيارات المفقودة فقط
        for tid in lost_ids:
            if tid in self._color_cache:
                del self._color_cache[tid]
            if tid in self._frame_counters:
                del self._frame_counters[tid]
            if tid in self._color_history:
                del self._color_history[tid]
            del self.last_seen_frame[tid]


    def reset(self):
        """تفريغ كلي لجميع القواميس"""
        self._color_history.clear()
        self._color_cache.clear()
        self._frame_counters.clear()
        self.last_seen_frame.clear()  # <--- [جديد]