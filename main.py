# # import os
# # import time

# # from detections import (
# #     CarDetection,
# #     CarTypeClassifier,
# #     CarColorDetection,
# #     CarMakeModelDetection,
# #     PlateDetector,
# #     YoloPlateDetector,
# # )


# # def load_mmr_labels(label_path):
# #     labels = []
# #     with open(label_path, "r", encoding="cp1251") as f:
# #         for line in f:
# #             line = line.strip()
# #             if line:
# #                 parts = line.split("\t")
# #                 make = parts[0]
# #                 model_name = parts[1] if len(parts) > 1 else ""
# #                 labels.append(f"{make} {model_name}".strip())
# #     return labels


# # def main():
# #     start_time = time.time()
    
# #     input_video_path = "input_videos/input_video11.mp4"
# #     output_video_path = "output_videos/output_video42.mp4"
# #     stub_path = "tracker_stubs/car_detection.pkl"
# #     read_from_stub = False  
# #     PLATE_BACKEND = "cv"    # "cv" أو "yolo"

# #     os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
# #     os.makedirs(os.path.dirname(stub_path), exist_ok=True)

# #     type_classifier = CarTypeClassifier(
# #         model_path="models/car_body_type_classifier.pt",
# #         class_map_path="models/idx_to_class.json",
# #         confidence_threshold=0.55,
# #         window_size=10,
# #         compute_interval=5,
# #     )

# #     color_detector = CarColorDetection(
# #         model_path="models/Car_Color_Detection.keras",
# #         history_size=15,
# #         rescale=1.0 / 255.0,
# #         min_confidence=0.55,
# #         compute_interval=5,
# #     )

# #     labels = load_mmr_labels("models/mmr-labels.txt")
# #     mmr_detector = CarMakeModelDetection(
# #         model_path="models/Car_MMR_Detection.mnn",
# #         class_names=labels,
# #         input_size=(128, 128),
# #         min_confidence=0.55,
# #         history_size=15,
# #         compute_interval=5,
# #     )

# #     if PLATE_BACKEND == "yolo":
# #         plate_detector = YoloPlateDetector(
# #             model_path="models/license-plate-finetune-v1n.pt",
# #         )
# #     else:
# #         plate_detector = PlateDetector()

# #     car_detector = CarDetection(
# #         model_path="models/yolo11n.pt",
# #         type_classifier=type_classifier,
# #         color_detector=color_detector,
# #         make_model_detector=mmr_detector,
# #         plate_detector=plate_detector,
# #         confidence_threshold=0.55,
# #     )

    
# #     print("Starting video processing stream...")
# #     car_detector.process_streaming(
# #         input_video_path=input_video_path,
# #         output_video_path=output_video_path,
# #         read_from_stub=read_from_stub,
# #         stub_path=stub_path,
# #     )

# #     car_detector.save_tracking_log("tracking_log35.json")

# #     end_time = time.time()
# #     elapsed_time = end_time - start_time
# #     minutes = int(elapsed_time // 60)
# #     seconds = elapsed_time % 60

# #     # car_detector.save_tracking_log("tracking_log_raw.json")
    
# #     final_report = car_detector.save_final_report(
# #         output_path="final_report.json",
# #         per_field_best=True   
# #     )

# #     print("=" * 40)
# #     print(" Done!")
# #     print(f" Total execution time: {minutes} m {seconds:.2f} s ({elapsed_time:.2f} seconds total)")
# #     print("=" * 40)


# # if __name__ == "__main__":
# #     main()


# import os
# import time
# import json

# from car_detection import CarDetection
# from car_color_detection import CarColorDetection
# from car_type_classifier import CarTypeClassifier
# from car_mmr_detection import CarMakeModelDetection
# from licence_plate_detection_algorithm import PlateDetector
# from yolo_licence_plate_detection import YoloPlateDetector

# from bridge.image_extractor import extract_best_vehicle_images
# from bridge.payload_builder import build_payload
# from bridge.callback import send_callback


# def load_mmr_labels(label_path):
#     labels = []
#     with open(label_path, "r", encoding="cp1251") as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 parts = line.split("\t")
#                 make = parts[0]
#                 model_name = parts[1] if len(parts) > 1 else ""
#                 labels.append(f"{make} {model_name}".strip())
#     return labels


# def init_detector(plate_backend="cv"):
#     type_classifier = CarTypeClassifier(
#         model_path="models/car_body_type_classifier.pt",
#         class_map_path="models/idx_to_class.json",
#         confidence_threshold=0.55,
#         window_size=10,
#         compute_interval=5,
#     )

#     color_detector = CarColorDetection(
#         model_path="models/Car_Color_Detection.keras",
#         history_size=15,
#         rescale=1.0 / 255.0,
#         min_confidence=0.55,
#         compute_interval=5,
#     )

#     labels = load_mmr_labels("models/mmr-labels.txt")
#     mmr_detector = CarMakeModelDetection(
#         model_path="models/Car_MMR_Detection.mnn",
#         class_names=labels,
#         input_size=(128, 128),
#         min_confidence=0.55,
#         history_size=15,
#         compute_interval=5,
#     )

#     if plate_backend == "yolo":
#         plate_detector = YoloPlateDetector(
#             model_path="models/license-plate-finetune-v1n.pt",
#         )
#     else:
#         plate_detector = PlateDetector()

#     return CarDetection(
#         model_path="models/yolo11n.pt",
#         type_classifier=type_classifier,
#         color_detector=color_detector,
#         make_model_detector=mmr_detector,
#         plate_detector=plate_detector,
#         confidence_threshold=0.55,
#     )


# def run_ai_job(video_id: int, input_video_path: str):
#     start_time = time.time()

#     storage_base = os.getenv("STORAGE_BASE_PATH", "C:/shared_storage")
#     plate_backend = os.getenv("PLATE_BACKEND", "cv")

#     output_video_path = os.path.join(storage_base, "processed", str(video_id), "processed.mp4")
#     os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

#     detector = init_detector(plate_backend)

#     print(f"[AI] ▶️  Starting video_id={video_id}")
#     detector.process_streaming(
#         input_video_path=input_video_path,
#         output_video_path=output_video_path,
#         read_from_stub=False,
#         stub_path=None,
#     )

#     final_report = detector.finalize_tracking_log(per_field_best=True)

#     vehicle_images = extract_best_vehicle_images(
#         input_video_path,
#         detector.tracking_log,
#         video_id,
#         storage_base
#     )

#     payload = build_payload(video_id, final_report, vehicle_images)
#     success = send_callback(payload)

#     # حفظ نسخة محلية
#     report_dir = os.path.join(storage_base, "reports")
#     os.makedirs(report_dir, exist_ok=True)
#     with open(os.path.join(report_dir, f"{video_id}.json"), "w", encoding="utf-8") as f:
#         json.dump(payload, f, ensure_ascii=False, indent=2)

#     elapsed = time.time() - start_time
#     print(f"[AI] ✅  Done video_id={video_id} | vehicles={len(final_report)} | time={elapsed:.1f}s | callback={'OK' if success else 'FAIL'}")


# if __name__ == "__main__":
#     run_ai_job(
#         video_id=3,
#         input_video_path="input_videos/input_video8.mp4"
#     )

import os
import time

from detections import (
    CarDetection,
    CarTypeClassifier,
    CarColorDetection,
    CarMakeModelDetection,
    PlateDetector,
    YoloPlateDetector,
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
    
    input_video_path = "input_videos/input_video8.mp4"
    output_video_path = "output_videos/output_video41.mp4"
    stub_path = "tracker_stubs/car_detection.pkl"
    read_from_stub = False
    PLATE_BACKEND = "cv"

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(stub_path), exist_ok=True)

    type_classifier = CarTypeClassifier(
        model_path="models/car_body_type_classifier.pt",
        class_map_path="models/idx_to_class.json",
        confidence_threshold=0.55,
        window_size=10,
        compute_interval=5,
    )

    color_detector = CarColorDetection(
        model_path="models/Car_Color_Detection.keras",
        history_size=15,
        rescale=1.0 / 255.0,
        min_confidence=0.55,
        compute_interval=5,
    )

    labels = load_mmr_labels("models/mmr-labels.txt")
    mmr_detector = CarMakeModelDetection(
        model_path="models/Car_MMR_Detection.mnn",
        class_names=labels,
        input_size=(128, 128),
        min_confidence=0.55,
        history_size=15,
        compute_interval=5,
    )

    if PLATE_BACKEND == "yolo":
        plate_detector = YoloPlateDetector(
            model_path="models/license-plate-finetune-v1n.pt",
        )
    else:
        plate_detector = PlateDetector()

    car_detector = CarDetection(
        model_path="models/yolo11n.pt",
        type_classifier=type_classifier,
        color_detector=color_detector,
        make_model_detector=mmr_detector,
        plate_detector=plate_detector,
        confidence_threshold=0.55,
    )

    print("Starting video processing stream...")
    car_detector.process_streaming(
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        read_from_stub=read_from_stub,
        stub_path=stub_path,
    )

    car_detector.save_tracking_log("tracking_log32.json")

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    final_report = car_detector.save_final_report(
        output_path="final_report.json",
        per_field_best=True
    )

    print("=" * 40)
    print(" Done!")
    print(f" Total execution time: {minutes} m {seconds:.2f} s ({elapsed_time:.2f} seconds total)")
    print("=" * 40)


if __name__ == "__main__":
    main()