# import os
# import time
# import json
# import subprocess
# import shutil

# from detections import (
#     CarDetection,
#     CarTypeClassifier,
#     CarColorDetection,
#     CarMakeModelDetection,
#     PlateDetector,
#     YoloPlateDetector,
# )

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


# def convert_to_h264(input_path: str, output_path: str) -> bool:
#     if not shutil.which("ffmpeg"):
#         print("[FFMPEG] WARNING: ffmpeg not found in PATH. Video may not play in browser.")
#         print("[FFMPEG] Download from: https://ffmpeg.org/download.html")
#         return False

#     temp_path = output_path.replace(".mp4", "_h264_temp.mp4")
#     try:
#         subprocess.run([
#             "ffmpeg", "-y", "-i", input_path,
#             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
#             "-c:a", "aac", "-b:a", "128k",
#             "-movflags", "+faststart",
#             "-pix_fmt", "yuv420p",
#             temp_path
#         ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
#         os.replace(temp_path, output_path)
#         print("[FFMPEG]  Converted to H.264 successfully")
#         return True
#     except subprocess.CalledProcessError as e:
#         print(f"[FFMPEG]  Conversion failed: {e}")
#         if os.path.exists(temp_path):
#             os.remove(temp_path)
#         return False
#     except Exception as e:
#         print(f"[FFMPEG]  Error: {e}")
#         if os.path.exists(temp_path):
#             os.remove(temp_path)
#         return False


# def run_ai_job(video_id: int, input_video_path: str):
#     start_time = time.time()

#     storage_base = os.getenv("STORAGE_BASE_PATH", "C:/shared_storage")
#     plate_backend = os.getenv("PLATE_BACKEND", "cv")

#     output_video_path = os.path.join(storage_base, "processed", str(video_id), "processed.mp4")
#     os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

#     detector = init_detector(plate_backend)

#     print(f"[AI]  Starting video_id={video_id}")
    
#     ai_start_time = time.time()
    
#     detector.process_streaming(
#         input_video_path=input_video_path,
#         output_video_path=output_video_path,
#         read_from_stub=False,
#         stub_path=None,
#     )
    
#     ai_processing_time = time.time() - ai_start_time
#     ai_mins = int(ai_processing_time // 60)
#     ai_secs = ai_processing_time % 60  
    
#     print(f"[AI]  Model processing finished in: {ai_mins} m {ai_secs:.2f} s ({ai_processing_time:.2f} seconds total)")

#     print("[AI]  Converting video to H.264...")
#     raw_path = output_video_path.replace(".mp4", "_raw.mp4")
#     os.rename(output_video_path, raw_path)
#     convert_to_h264(raw_path, output_video_path)
#     if os.path.exists(raw_path):
#         os.remove(raw_path)

#     final_report = detector.finalize_tracking_log(per_field_best=True)

#     vehicle_images = extract_best_vehicle_images(
#         input_video_path,
#         detector.tracking_log,
#         video_id,
#         storage_base
#     )

#     payload = build_payload(video_id, final_report, vehicle_images)
    
#     payload["processing_time_seconds"] = round(ai_processing_time, 2)
#     payload["processing_time_formatted"] = f"{ai_mins} m {ai_secs:.2f} s"
    
#     success = send_callback(payload)

#     report_dir = os.path.join(storage_base, "reports")
#     os.makedirs(report_dir, exist_ok=True)
#     with open(os.path.join(report_dir, f"{video_id}.json"), "w", encoding="utf-8") as f:
#         json.dump(payload, f, ensure_ascii=False, indent=2)

#     total_elapsed = time.time() - start_time
#     tot_mins = int(total_elapsed // 60)
#     tot_secs = total_elapsed % 60
    
#     print(f"[AI]   Done video_id={video_id} | vehicles={len(final_report)} | AI time: {ai_mins}m {ai_secs:.2f}s ({ai_processing_time:.2f}s total) | Total time: {tot_mins}m {tot_secs:.2f}s ({total_elapsed:.2f}s total) | callback={'OK' if success else 'FAIL'}")


import os
import time
import json

from detections import (
    CarDetection,
    CarTypeClassifier,
    CarColorDetection,
    CarMakeModelDetection,
    PlateDetector,
    YoloPlateDetector,
)

from bridge.image_extractor import extract_best_vehicle_images
from bridge.payload_builder import build_payload
from bridge.callback import send_callback


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


def init_detector(plate_backend="cv"):
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

    if plate_backend == "yolo":
        plate_detector = YoloPlateDetector(
            model_path="models/license-plate-finetune-v1n.pt",
        )
    else:
        plate_detector = PlateDetector()

    return CarDetection(
        model_path="models/yolo11n.pt",
        type_classifier=type_classifier,
        color_detector=color_detector,
        make_model_detector=mmr_detector,
        plate_detector=plate_detector,
        confidence_threshold=0.55,
    )


def run_ai_job(video_id: int, input_video_path: str):
    start_time = time.time()

    storage_base = os.getenv("STORAGE_BASE_PATH", "C:/shared_storage")
    plate_backend = os.getenv("PLATE_BACKEND", "cv")

    output_video_path = os.path.join(storage_base, "processed", str(video_id), "processed.mp4")
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    detector = init_detector(plate_backend)

    print(f"[AI]  Starting video_id={video_id}")
    
    ai_start_time = time.time()
    
    detector.process_streaming(
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        read_from_stub=False,
        stub_path=None,
    )
    
    ai_processing_time = time.time() - ai_start_time
    ai_mins = int(ai_processing_time // 60)
    ai_secs = ai_processing_time % 60  
    
    print(f"[AI]  Model processing finished in: {ai_mins} m {ai_secs:.2f} s ({ai_processing_time:.2f} seconds total)")
    # الفيديو خرج مباشرة بصيغة H.264 من FFmpegStreamWriter داخل process_streaming،
    # ما في داعي لمرحلة تحويل/إعادة ترميز منفصلة بعد الآن.

    final_report = detector.finalize_tracking_log(per_field_best=True)

    vehicle_images = extract_best_vehicle_images(
        input_video_path,
        detector.tracking_log,
        video_id,
        storage_base
    )

    payload = build_payload(video_id, final_report, vehicle_images)
    
    payload["processing_time_seconds"] = round(ai_processing_time, 2)
    payload["processing_time_formatted"] = f"{ai_mins} m {ai_secs:.2f} s"
    
    success = send_callback(payload)

    report_dir = os.path.join(storage_base, "reports")
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, f"{video_id}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - start_time
    tot_mins = int(total_elapsed // 60)
    tot_secs = total_elapsed % 60
    
    print(f"[AI]   Done video_id={video_id} | vehicles={len(final_report)} | AI time: {ai_mins}m {ai_secs:.2f}s ({ai_processing_time:.2f}s total) | Total time: {tot_mins}m {tot_secs:.2f}s ({total_elapsed:.2f}s total) | callback={'OK' if success else 'FAIL'}")