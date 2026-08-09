import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2 as cv
from collections import deque
import numpy as np


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

    # ===============================
    # BATCHED FRAME-LEVEL CLASSIFICATION
    # ===============================
    def classify_and_vote_for_frame(self, frame, car_dict, cls_name_dict, frame_idx):
        """
        تصنيف نوع جسم السيارة بشكل مجمع لكل الإطار.
        car_dict: {track_id: bbox, ...}
        cls_name_dict: {track_id: cls_name, ...} (للتوافق مع الاستخدام الحالي)
        """
        if not car_dict:
            return {}

        needs_compute = []
        cached_results = {}

        for track_id in car_dict.keys():
            self._frame_counters[track_id] = self._frame_counters.get(track_id, 0) + 1
            self.last_seen_frame[track_id] = frame_idx

            current_count = self._frame_counters[track_id]
            if track_id in self._type_cache and current_count % self.compute_interval != 0:
                cached = self._type_cache[track_id]
                cached_results[track_id] = (cached["final_type"], cached["final_conf"])
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
    def classify_and_vote(self, crop, track_id, cls_name, frame_idx):
        """للتوافق مع الاستدعاءات القديمة — تستدعي النسخة المجمعة"""
        if track_id == -1:
            return self.predict(crop)

        h, w = crop.shape[:2]
        bbox = [0, 0, w, h]
        cls_dict = {track_id: cls_name}
        result = self.classify_and_vote_for_frame(crop, {track_id: bbox}, cls_dict, frame_idx)
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
        """تبني crop واحد لسيارة واحدة (PyTorch tensor)."""
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

        # تجميع الـ tensors بـ batch واحد
        batch = torch.stack(per_car_crops, dim=0).to(self.device)

        with torch.no_grad():
            output = self.model(batch)
            probs = torch.softmax(output, dim=1)

        predictions = probs.cpu().numpy()

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
            raw_label = self.idx_to_class.get(best_idx, "Unknown")

            if confidence < self.confidence_threshold:
                raw_results[track_id] = ("Unknown", 0.0)
            else:
                raw_results[track_id] = (raw_label, confidence)

        return raw_results

    # ===============================
    # TEMPORAL SMOOTHING (Weighted Voting)
    # ===============================
    def _update_history_and_vote(self, raw_results):
        """تحديث التاريخ والتصويت الزمني لكل السيارات."""
        stable_types = {}

        for track_id, (label, confidence) in raw_results.items():
            if track_id not in self.vote_buffer:
                self.vote_buffer[track_id] = deque(maxlen=self.window_size)

            if label != "Unknown":
                self.vote_buffer[track_id].append((label, confidence))

            buffer = self.vote_buffer[track_id]

            if not buffer:
                stable_types[track_id] = ("Unknown", 0.0)
                continue

            # حساب التصويت الموزون
            weighted_scores = {}
            for lbl, conf in buffer:
                weighted_scores[lbl] = weighted_scores.get(lbl, 0.0) + conf

            smoothed_label = max(weighted_scores, key=weighted_scores.get)
            total_weight = sum(weighted_scores.values())
            vote_ratio = weighted_scores[smoothed_label] / total_weight

            # تطبيق المنطق الذكي الموحد
            if vote_ratio >= 0.5:
                final_type = smoothed_label
                final_conf = vote_ratio
            elif label != "Unknown" and confidence >= self.confidence_threshold:
                final_type = label
                final_conf = confidence
            else:
                final_type = "Unknown"
                final_conf = 0.0

            # تحديث الكاش
            self._type_cache[track_id] = {
                "final_type": final_type,
                "final_conf": final_conf
            }

            stable_types[track_id] = (final_type, final_conf)

        return stable_types

    # ===============================
    # SINGLE PREDICTION (Backward compatibility)
    # ===============================
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
        pil_img = Image.fromarray(crop_rgb)
        return self.transform(pil_img)

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