from utils import read_video, save_video
from detections import CarDetection, CarTypeClassifier, CarColorDetection
import time

def main():
    start_time = time.time()
    input_video_path = 'input_videos/input_video3.mp4'

    video_frames, fps = read_video(input_video_path)

    # 1. تمرير window_size و confidence_threshold الخاصة بالتصويت هنا
    type_classifier = CarTypeClassifier(
        model_path='models/car_body_type_classifier.pt',
        class_map_path='models/idx_to_class.json',
        confidence_threshold=0.6,
        window_size=10
    )

    color_detector = CarColorDetection(
        model_path='Car_Color_Detection.keras',
        history_size=15,
        rescale=1.0 / 255.0,
        min_confidence=0.40,
    )

    # 2. CarDetection لم يعد يحتاج window_size
    car_detector = CarDetection(
        model_path='yolo11n.pt',
        type_classifier=type_classifier,
        color_detector=color_detector,
        confidence_threshold=0.6
    )

    car_detections = car_detector.detect_frames(
        video_frames,
        read_from_stub=False,
        stub_path="tracker_stubs/car_detection.pkl"
    )

    output_frames = car_detector.draw_bboxes(video_frames, car_detections)
    save_video(output_frames, output_path="output_videos/output_video26.mp4", fps=fps)

    car_detector.save_tracking_log("tracking_log21.json")

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    print("=" * 40)
    print(f" Done!")
    print(f" Total execution time: {minutes} m {seconds:.2f} s ({elapsed_time:.2f} seconds total)")
    print("=" * 40)

if __name__ == "__main__":
    main()