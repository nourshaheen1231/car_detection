from ultralytics import YOLO
import cv2 as cv

model = YOLO('yolo11n.pt')

test_images = [
    # 'test_images/service10.jpg',
    'test_images/hyundai11.jpg'
]

for img_path in test_images:
    frame = cv.imread(img_path)
    if frame is None:
        print(f"can not open photo: {img_path}")
        continue

    results = model.predict(frame, iou=0.1, conf=0.25)[0]
    id_name_dict = results.names

    print(f"\n=== photo result: {img_path} ===")
    if len(results.boxes) == 0:
        print("  No objects detected")

    for box in results.boxes:
        cls_id = int(box.cls.tolist()[0])
        cls_name = id_name_dict[cls_id]
        confidence = float(box.conf.tolist()[0])
        bbox = box.xyxy.tolist()[0]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        print(f"  class: {cls_name:12s} | confidence: {confidence:.3f} | box size: {int(width)}x{int(height)}")
