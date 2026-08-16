import cv2
import os

def extract_best_vehicle_images(video_path: str, tracking_log: dict, video_id: int, storage_base: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[EXTRACTOR] Failed to open video")
        return {}

    out_dir = os.path.join(storage_base, "vehicles", str(video_id))
    os.makedirs(out_dir, exist_ok=True)

    images = {}
    for track_id, frames in tracking_log.items():
        if not frames:
            continue
        
        best_frame_data = max(frames, key=lambda f: f.get("yolo_confidence", 0.0))
        frame_num = best_frame_data["frame"]
        bbox = best_frame_data.get("bbox")
        
        if not bbox:
            print(f"[EXTRACTOR] No bbox for track {track_id}, skipping")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        crop = frame[y1:y2, x1:x2]
        
        filename = f"track_{track_id}.jpg"
        filepath = os.path.join(out_dir, filename)
        cv2.imwrite(filepath, crop)
        images[track_id] = f"vehicles/{video_id}/{filename}"

    cap.release()
    print(f"[EXTRACTOR] Saved {len(images)} vehicle crops")
    return images