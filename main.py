from utils import read_video, save_video
from utils.profiler import PerformanceProfiler
from detections import CarDetection, CarColorDetection


def main():
    input_video_path = 'input_videos/input_video2.mp4'

    # Read video frames
    video_frames = read_video(input_video_path)

    profiler = PerformanceProfiler()

    car_detecttor = CarDetection(model_path='yolo11n.pt', profiler=profiler)

    # Initialize color detection using the pretrained CNN model.
    color_detector = CarColorDetection(
        model_path='Car_Color_Detection.keras',
        history_size=15,
        rescale=1.0 / 255.0,
        min_confidence=0.40,
        profiler=profiler,
    )

    output_frames = car_detecttor.process_video(
        video_frames,
        color_detector=color_detector,
        read_from_stub=False,
        stub_path="tracker_stubs/car_detection.pkl",
    )

    # Save processed video
    save_video(output_frames, output_path="output_videos/output_video22222.mp4")

    profiler.print_summary()


if __name__ == "__main__":
    main()