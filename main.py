from utils import read_video, save_video
from detections import CarDetection

def main():
    input_video_path = 'input_videos/input_video2.mp4'
    
    # Read video frames
    video_frames = read_video(input_video_path)

    # Initialize car detection
    car_detecttor = CarDetection(model_path='yolo11n.pt')

    # Detect cars in video frames
    car_detections = car_detecttor.detect_frames(video_frames,read_from_stub=False,stub_path="tracker_stubs/car_detection.pkl")

    # Draw bounding boxes on video frames
    output_frames = car_detecttor.draw_bboxes(video_frames, car_detections)

    # Save processed video
    save_video(output_frames, output_path="output_videos/output_video.mp4")

if __name__ == "__main__":
    main()