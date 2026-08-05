import cv2 as cv
from detections import CarDetection, CarTypeClassifier, CarColorDetection

def test_single_images():
    test_images = [
        # 'test_images/hyundai6.jpg'
        # 'test_images/sport2.jpg'
        'test_images/service3.jpg',
        # 'test_images/bus4.jpg'
    ]

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

    car_detector = CarDetection(
        model_path='yolo11n.pt',
        type_classifier=type_classifier,
        color_detector=color_detector,
        confidence_threshold=0.6,
    )

    for img_path in test_images:
        frame = cv.imread(img_path)
        if frame is None:
            print(f"can not open photo: {img_path}")
            continue

        car_list = car_detector.detect_frame(frame, frame_idx=0)

        print(f"\n=== photo result: {img_path} ===")
        if not car_list:
            print("  No cars detected in the image")

        for detection in car_list:
            print(f"  {detection['yolo_class']} | Type: {detection['type_name']} | Color: {detection['color_name']} (track_id={detection['track_id']})")

        output_frame = car_detector.draw_bboxes([frame], [car_list])[0]
        output_path = img_path.replace('test_images/', 'test_images/result_color_')
        cv.imwrite(output_path, output_frame)
        print(f"  → photo saved in: {output_path}")

if __name__ == "__main__":
    test_single_images()