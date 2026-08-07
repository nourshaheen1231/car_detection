import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2 as cv
from collections import deque

class CarTypeClassifier:
    def __init__(self, model_path, class_map_path, confidence_threshold=0.55, window_size=10, compute_interval=5, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.window_size = window_size

        self.compute_interval = compute_interval
        self._type_cache = {}
        self._frame_counters = {}
        self.vote_buffer = {}

        self.last_seen_frame = {}
        self.max_lost_frames = 120

        with open(class_map_path, "r", encoding="utf-8") as f:
            idx_to_class = json.load(f)
        self.idx_to_class = {int(k): v for k, v in idx_to_class.items()}

        # FIX: infer num_classes from checkpoint, not JSON
        state_dict = torch.load(model_path, map_location=self.device)
        num_classes = state_dict['classifier.1.weight'].shape[0]

        if num_classes != len(self.idx_to_class):
            print(f"[WARNING] Model has {num_classes} classes but JSON has {len(self.idx_to_class)}. Using model's count.")

        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        self.model.load_state_dict(state_dict)
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

    def predict(self, crop):
        if crop is None or crop.size == 0:
            return "Unknown", 0.0

        crop_rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

        confidence = confidence.item()
        raw_label = self.idx_to_class.get(predicted.item(), "Unknown")

        return raw_label, confidence

    def classify_and_vote(self, crop, track_id, cls_name, frame_idx):
        """
        crop: crop جاهز
        track_id: id التتبع
        cls_name: صنف YOLO (car/truck) - مش مستخدم حالياً بس موجود للتوافق
        frame_idx: رقم الإطار
        returns: (final_type, final_conf)
        """
        if track_id == -1:
            return self.predict(crop)

        self._frame_counters[track_id] = self._frame_counters.get(track_id, 0) + 1
        current_count = self._frame_counters[track_id]

        self.last_seen_frame[track_id] = frame_idx

        # 1. التحقق من الـ Cache
        if track_id in self._type_cache:
            if current_count % self.compute_interval != 0:
                c = self._type_cache[track_id]
                return c["final_type"], c["final_conf"]

        # 2. إجراء التنبؤ (عملية ثقيلة - تحدث كل 5 فريمات مثلاً)
        raw_label, confidence = self.predict(crop)

        # 3. التصويت الزمني
        smoothed_label, vote_ratio = self._vote(track_id, raw_label, confidence)

        # --- تطبيق منطقك الذكي ---
        if vote_ratio >= 0.5: 
            # الحالة الأولى: التصويت ناجح والأغلبية متفقة
            final_type = smoothed_label
            final_conf = vote_ratio
        elif raw_label != "Unknown" and confidence >= self.confidence_threshold:
            # الحالة الثانية: التصويت فاشل، لكن الفريم الحالي وااااضح جداً
            final_type = raw_label
            final_conf = confidence
        else:
            # الحالة الثالثة: التصويت فاشل والفريم الحالي ضعيف، هون نستسلم
            final_type = "Unknown"
            final_conf = 0.0

        # تحديث الكاش بالقيم النهائية
        self._type_cache[track_id] = {
            "final_type": final_type,
            "final_conf": final_conf
        }

        return final_type, final_conf

    def _vote(self, track_id, raw_label, confidence):
        # إنشاء ذاكرة بحجم محدد تلقائياً للسيارة عند ظهورها أول مرة
        if track_id not in self.vote_buffer:
            self.vote_buffer[track_id] = deque(maxlen=self.window_size)

        buffer = self.vote_buffer[track_id]

        # 1. إضافة التنبؤ للسجل فقط إذا لم يكن Unknown
        if raw_label != "Unknown":
            buffer.append((raw_label, confidence))

        # 2. حماية: إذا كان السجل فارغاً تماماً
        if not buffer:
            return "Unknown", 0.0

        # 3. حساب التصويت الموزون
        weighted_scores = {}
        for label, conf in buffer:
            weighted_scores[label] = weighted_scores.get(label, 0.0) + conf

        smoothed_label = max(weighted_scores, key=weighted_scores.get)
        total_weight = sum(weighted_scores.values())
        vote_ratio = weighted_scores[smoothed_label] / total_weight

        return smoothed_label, vote_ratio

    def cleanup_inactive_tracks(self, current_frame_idx):
        """
        يحذف بيانات السيارات التي غابت عن الشاشة لفترة تتجاوز max_lost_frames
        """
        lost_ids = []

        for tid, last_frame in list(self.last_seen_frame.items()):
            if (current_frame_idx - last_frame) > self.max_lost_frames:
                lost_ids.append(tid)

        # تنظيف شامل للسيارات المفقودة فقط
        for tid in lost_ids:
            if tid in self._type_cache:
                del self._type_cache[tid]
            if tid in self._frame_counters:
                del self._frame_counters[tid]
            if tid in self.vote_buffer:
                del self.vote_buffer[tid]
            del self.last_seen_frame[tid]

    def reset(self):
        """تفريغ كلي لجميع القواميس عند إعادة تشغيل الفيديو"""
        self._type_cache.clear()
        self._frame_counters.clear()
        self.vote_buffer.clear()
        self.last_seen_frame.clear()