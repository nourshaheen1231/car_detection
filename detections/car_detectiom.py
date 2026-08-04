import cv2 as cv
import pickle
from ultralytics import YOLO

class CarDetection():
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def process_video(
        self,
        video_frames,
        color_detector=None,
        make_model_detector=None,
        read_from_stub=False,
        stub_path=None,
    ):
        output_frames = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                car_detections = pickle.load(f)

            for frame, car_dict in zip(video_frames, car_detections):
                output_frames.append(
                    self._draw_frame(frame, car_dict, color_detector, make_model_detector)
                )
            return output_frames

        all_car_dicts = [] if stub_path is not None else None

        for frame in video_frames:
            car_dict = self.detect_frame(frame)
            if all_car_dicts is not None:
                all_car_dicts.append(car_dict)

            output_frames.append(
                self._draw_frame(frame, car_dict, color_detector, make_model_detector)
            )

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(all_car_dicts, f)

        return output_frames

    def _draw_frame(self, frame, car_dict, color_detector=None, make_model_detector=None):
        for track_id, bbox in car_dict.items():
            x1, y1, x2, y2 = bbox

            labels = [f"Car #{track_id}"]

            if color_detector is not None:
                color_name = color_detector.get_stable_color(track_id, frame, bbox)
                color_conf = color_detector._color_cache.get(track_id, {}).get("confidence", 0)
                labels.append(f"{color_name} ({color_conf:.2f})")

            if make_model_detector is not None:
                mm_name = make_model_detector.get_make_model(track_id, frame, bbox)
                mm_conf = make_model_detector._cache.get(track_id, {}).get("confidence", 0)
                labels.append(f"{mm_name} ({mm_conf:.2f})")

            label_text = " - ".join(labels)

            cv.putText(
                frame,
                label_text,
                (int(x1), int(y1 - 10)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 255, 0),
                2,
            )
        return frame

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                car_detections = pickle.load(f)
            return car_detections

        car_detections = []
        for frame in frames:
            car_dict = self.detect_frame(frame)
            car_detections.append(car_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(car_detections, f)

        return car_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, iou=0.1, conf=0.3, persist=True)[0]
        id_name_dict = results.names

        car_dict = {}
        for box in results.boxes:
            cls_id = int(box.cls.tolist()[0])
            cls_name = id_name_dict[cls_id]
            if cls_name != "car":
                continue

            if box.id is None:
                continue

            track_id = int(box.id.tolist()[0])
            bbox = box.xyxy.tolist()[0]
            car_dict[track_id] = bbox

        return car_dict

    def draw_bboxes(self, video_frames, car_detections, color_detector=None):
      
        output_frames = []
        for frame, car_dict in zip(video_frames, car_detections):
            output_frames.append(self._draw_frame(frame, car_dict, color_detector))
        return output_frames