import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2 as cv

class CarTypeClassifier:
    def __init__(self, model_path, class_map_path, confidence_threshold=0.6, window_size=10, compute_interval=5, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.window_size = window_size
        
        # === المتغيرات الجديدة الخاصة بالـ Cache ===
        self.compute_interval = compute_interval
        self._type_cache = {}
        self._frame_counters = {}
        # مخزن التصويت لكل سيارة حسب الـ track_id
        self.vote_buffer = {}

        with open(class_map_path, "r", encoding="utf-8") as f:
            idx_to_class = json.load(f)
        self.idx_to_class = {int(k): v for k, v in idx_to_class.items()}
        num_classes = len(self.idx_to_class)

        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, frame, bbox):
        # (نفس الكود القديم تماماً بدون تعديل)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return "Unknown", 0.0

        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

        confidence = confidence.item()
        raw_label = self.idx_to_class[predicted.item()]

        return raw_label, confidence

    def classify_and_vote(self, frame, bbox, track_id, cls_name):
        """
        تقوم باستدعاء التنبؤ ثم تطبيق نظام التصويت الزمني والكاش
        """
        self._frame_counters[track_id] = self._frame_counters.get(track_id, 0) + 1
        current_count = self._frame_counters[track_id]

        # 1. التحقق من الـ Cache
        if track_id in self._type_cache:
            if current_count % self.compute_interval != 0:
                # إرجاع القيم المحفوظة مسبقاً من الكاش
                c = self._type_cache[track_id]
                return c["label_str"], c["raw_label"], c["confidence"], c["smoothed_label"], c["vote_ratio"]
        # 2. إجراء التنبؤ (عملية ثقيلة - تحدث كل 5 فريمات مثلاً)
        raw_label, confidence = self.predict(frame, bbox)

        if track_id == -1:
            display_type = raw_label if confidence >= self.confidence_threshold else "Unknown"
            label_str = f"{cls_name} - {display_type} ({confidence:.2f})"
            return label_str, raw_label, confidence, raw_label, 0.0 

        # 3. التصويت الزمني
        smoothed_label, vote_ratio = self._vote(track_id, raw_label, confidence)
        display_type = smoothed_label if vote_ratio >= 0.5 else "Unknown"
        label_str = f"{cls_name} - {display_type} ({confidence:.2f})"

        # 4. حفظ النتائج في الـ Cache للاستخدام في الفريمات القادمة
        self._type_cache[track_id] = {
            "label_str": label_str,
            "raw_label": raw_label,
            "confidence": confidence,
            "smoothed_label": smoothed_label,
            "vote_ratio": vote_ratio
        }

        return label_str, raw_label, confidence, smoothed_label, vote_ratio

    def _vote(self, track_id, raw_label, confidence):
        # (نفس الكود القديم تماماً بدون تعديل)
        if track_id not in self.vote_buffer:
            self.vote_buffer[track_id] = []

        buffer = self.vote_buffer[track_id]
        buffer.append((raw_label, confidence))
        if len(buffer) > self.window_size:
            buffer.pop(0)

        weighted_scores = {}
        for label, conf in buffer:
            weighted_scores[label] = weighted_scores.get(label, 0) + conf

        smoothed_label = max(weighted_scores, key=weighted_scores.get)
        total_weight = sum(weighted_scores.values())
        vote_ratio = weighted_scores[smoothed_label] / total_weight
        return smoothed_label, vote_ratio


    def cleanup_inactive_tracks(self, active_track_ids):
        """
        تستقبل قائمة الـ track_ids التابعة للسيارات الموجودة حالياً في الفريم،
        وتحذف بيانات أي سيارة اختفت لمنع استهلاك الذاكرة.
        """
        active_set = set(active_track_ids)

        # 1. تنظيف الكاش
        for tid in list(self._type_cache.keys()):
            if tid not in active_set:
                del self._type_cache[tid]

        # 2. تنظيف عداد الفريمات
        for tid in list(self._frame_counters.keys()):
            if tid not in active_set:
                del self._frame_counters[tid]

        # 3. تنظيف بافر التصويت
        for tid in list(self.vote_buffer.keys()):
            if tid not in active_set:
                del self.vote_buffer[tid]

    def reset(self):
        """تفريغ كلي لجميع القواميس عند إعادة تشغيل الفيديو"""
        self._type_cache.clear()
        self._frame_counters.clear()
        self.vote_buffer.clear()        


        