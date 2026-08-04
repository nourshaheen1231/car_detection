from utils import read_video, save_video
from detections import CarDetection, CarColorDetection, CarMakeModelDetection


def main():
    input_video_path = "input_videos/drive1.mp4"

    # Read video frames
    video_frames = read_video(input_video_path)

    # Initialize car detection
    car_detector = CarDetection(model_path="yolo11n.pt")

    # Initialize color detection
    color_detector = CarColorDetection(
        model_path="Car_Color_Detection.keras",
        history_size=15,
        rescale=1.0 / 255.0,
        min_confidence=0.40,
    )

    label_path = "mmr-labels.txt"
    labels = []
    with open("mmr-labels.txt", "r", encoding="cp1251") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("\t")
                make = parts[0]
                model_name = parts[1] if len(parts) > 1 else ""
                labels.append(f"{make} {model_name}".strip())

    mm_detector = CarMakeModelDetection(
        model_path="Car_MMR_Detection.mnn",
        class_names=labels,
        input_size=(128, 128),
        min_confidence=0.4
    )

    # Process video through combined pipeline
    output_frames = car_detector.process_video(
        video_frames,
        color_detector=color_detector,
        make_model_detector=mm_detector,
        read_from_stub=False,
        stub_path="tracker_stubs/car_detection.pkl",
    )

    # Save processed video
    save_video(output_frames, output_path="output_videos/output_video2.mp4")


if __name__ == "__main__":
    main()