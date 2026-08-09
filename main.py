import os
import pickle
import time

from utils import iter_video_frames, get_video_fps, VideoWriterContext
from detections import (
    CarDetection,
    CarTypeClassifier,
    CarColorDetection,
    CarMakeModelDetection,
    PlateDetector,
)


def load_mmr_labels(label_path):
    labels = []
    with open(label_path, "r", encoding="cp1251") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("\t")
                make = parts[0]
                model_name = parts[1] if len(parts) > 1 else ""
                labels.append(f"{make} {model_name}".strip())
    return labels


def main():
    start_time = time.time()
    input_video_path = "input_videos/input_video4.mp4"
    output_video_path = "output_videos/output_video38.mp4"
    stub_path = "tracker_stubs/car_detection.pkl"
    read_from_stub = False  # خليها True لو بدك تعيد الرسم بس من نتائج مخزّنة سابقاً بدون إعادة الكشف

    fps = get_video_fps(input_video_path)

    # 1. تهيئة مصنف نوع جسم السيارة
    type_classifier = CarTypeClassifier(
        model_path="models/car_body_type_classifier.pt",
        class_map_path="models/idx_to_class.json",
        confidence_threshold=0.55,
        window_size=10,
        compute_interval=5,
    )

    # 2. تهيئة مكتشف لون السيارة
    color_detector = CarColorDetection(
        model_path="models/Car_Color_Detection.keras",
        history_size=15,
        rescale=1.0 / 255.0,
        min_confidence=0.55,
        compute_interval=5,
    )

    # 3. تهيئة مكتشف ماركة وموديل السيارة (MMR)
    labels = load_mmr_labels("models/mmr-labels.txt")
    mmr_detector = CarMakeModelDetection(
        model_path="models/Car_MMR_Detection.mnn",
        class_names=labels,
        input_size=(128, 128),
        min_confidence=0.55,
        history_size=15,
        compute_interval=5,
    )

    # 4. تهيئة الكاشف الرئيسي وتمرير النماذج الثلاثة + كاشف اللوحة
    car_detector = CarDetection(
        model_path="models/yolo11n.pt",
        type_classifier=type_classifier,
        color_detector=color_detector,
        make_model_detector=mmr_detector,
        plate_detector=PlateDetector(),
        confidence_threshold=0.55,
    )

    # =====================================================
    # تحميل نتائج كشف مخزّنة مسبقاً (اختياري) بدل إعادة تشغيل
    # الموديلات على كل فريم من جديد
    # =====================================================
    cached_car_detections = None
    if read_from_stub and os.path.exists(stub_path):
        with open(stub_path, "rb") as f:
            cached_car_detections = pickle.load(f)

    # =====================================================
    # المعالجة الفعلية: فريم فريم (streaming)
    # ما في ولا لحظة فيها الفيديو كامل محمّل بالذاكرة
    # =====================================================
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(stub_path), exist_ok=True)

    all_car_detections = []  # بيانات خفيفة بس (bboxes/labels)، مش صور — حجمها مهمل
    writer = None
    frame_idx = 0

    for frame in iter_video_frames(input_video_path):
        if writer is None:
            h, w = frame.shape[:2]
            writer = VideoWriterContext(output_video_path, fps, (w, h))

        if cached_car_detections is not None:
            car_list = cached_car_detections[frame_idx]
        else:
            # detect_frame بتعمل batching داخلياً لكل الموديلات
            car_list = car_detector.detect_frame(frame, frame_idx)
            all_car_detections.append(car_list)

        car_detector.draw_frame(frame, car_list)  # رسم بالمكان (in-place)
        writer.write(frame)

        frame_idx += 1
        # ملاحظة: ما في أي list فيها الفريمات نفسها — frame بيتحرر
        # من الذاكرة تلقائياً بعد كل تكرار

    if writer is not None:
        writer.release()

    # نخزّن نتائج الكشف (بيانات خفيفة) لو ما كنا عم نقرأ من stub أصلاً
    if cached_car_detections is None:
        with open(stub_path, "wb") as f:
            pickle.dump(all_car_detections, f)

    car_detector.save_tracking_log("tracking_log32.json")

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    print("=" * 40)
    print(" Done!")
    print(f" Total frames processed: {frame_idx}")
    print(f" Total execution time: {minutes} m {seconds:.2f} s ({elapsed_time:.2f} seconds total)")
    print("=" * 40)


if __name__ == "__main__":
    main()