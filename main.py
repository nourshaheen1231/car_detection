import time
from utils import read_video, save_video
from detections import (
    CarDetection,
    CarTypeClassifier,
    CarColorDetection,
    CarMakeModelDetection,
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
    input_video_path = "input_videos/input_video3.mp4"

    video_frames, fps = read_video(input_video_path)

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

    # 4. تهيئة الكاشف الرئيسي وتمرير النماذج الثلاثة
    car_detector = CarDetection(
        model_path="models/yolo11n.pt",
        type_classifier=type_classifier,
        color_detector=color_detector,
        make_model_detector=mmr_detector,
        confidence_threshold=0.55,
    )

    # معالجة الفريمات والتتبع
    car_detections = car_detector.detect_frames(
        video_frames,
        read_from_stub=False,
        stub_path="tracker_stubs/car_detection.pkl",
    )

    # رسم النتائج وحفظ الفيديو والسجلات
    output_frames = car_detector.draw_bboxes(video_frames, car_detections)
    save_video(output_frames, output_path="output_videos/output_video32.mp4", fps=fps)

    car_detector.save_tracking_log("tracking_log27.json")

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    print("=" * 40)
    print(" Done!")
    print(f" Total execution time: {minutes} m {seconds:.2f} s ({elapsed_time:.2f} seconds total)")
    print("=" * 40)


if __name__ == "__main__":
    main()

