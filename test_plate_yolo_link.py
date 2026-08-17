"""
تيست مرئي: كشف لوحة YOLO + OCR + ربطها بالسيارة + قياس الأزمنة.
  أخضر  = PROJECT (موقع اللوحة بدون موديل)
  وردي  = MODEL   (موديل موقع اللوحة اشتغل)
  النص المقروء وثقة كشف موقع اللوحة تظهر فوق بوكس اللوحة
"""

import os
import time
from collections import defaultdict

import cv2 as cv

from detections import CarDetection, YoloPlateDetector
from detections.car_detection import VEHICLE_CLASSES
from utils.video_utils import VideoWriterContext

INPUT_VIDEO = "input_videos/input_video2.mp4"
OUTPUT_VIDEO = "output_videos/plate_yolo_ocr_test.mp4"
OUTPUT_DIR = "debug_plate_yolo_ocr"
SHOW_WINDOWS = False
SAVE_FRAME_IMAGES = False
CONF = 0.55

COLOR_CAR = (0, 255, 255)
COLOR_PROJECT = (0, 220, 0)
COLOR_MODEL = (255, 0, 255)
COLOR_OCR_OK = (0, 255, 0)
COLOR_OCR_WAIT = (0, 200, 255)


def draw_banner(img, lines, bg_color):
    x, y = 10, 10
    pad = 8
    font = cv.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.65, 2
    sizes = [cv.getTextSize(t, font, scale, thick)[0] for t in lines]
    box_w = max(w for w, h in sizes) + pad * 2
    line_h = max(h for w, h in sizes) + 10
    box_h = line_h * len(lines) + pad * 2

    overlay = img.copy()
    cv.rectangle(overlay, (x, y), (x + box_w, y + box_h), bg_color, -1)
    cv.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    for i, text in enumerate(lines):
        ty = y + pad + (i + 1) * line_h - 6
        cv.putText(img, text, (x + pad, ty), font, scale, (255, 255, 255), thick, cv.LINE_AA)


def detect_cars_only(car_detector, frame):
    frame_h, frame_w = frame.shape[:2]
    results = car_detector.model.track(
        frame,
        persist=True,
        iou=0.1,
        conf=car_detector.confidence_threshold,
        verbose=False,
    )[0]

    car_metadata = {}
    id_name_dict = results.names
    for box in results.boxes:
        cls_id = int(box.cls.tolist()[0])
        cls_name = id_name_dict[cls_id]
        if cls_name not in VEHICLE_CLASSES:
            continue
        track_id = int(box.id.item()) if box.id is not None else -1
        if track_id == -1:
            continue
        bbox = box.xyxy.tolist()[0]
        crop = car_detector._crop(frame, bbox)
        if crop is None:
            continue
        car_metadata[track_id] = {
            "bbox": bbox,
            "cls_name": cls_name,
            "yolo_conf": float(box.conf.item()) if box.conf is not None else 0.0,
            "crop": crop,
        }
    return car_metadata, (frame_w, frame_h)


def main():
    if not os.path.exists(INPUT_VIDEO):
        raise FileNotFoundError(f"الفيديو غير موجود: {INPUT_VIDEO}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    cap = cv.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {INPUT_VIDEO}")
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)

    t0 = time.perf_counter()
    plate_detector = YoloPlateDetector(
        model_path="models/license-plate-finetune-v1n.pt",
        enable_ocr=True,
        ocr_retry_interval=10,
    )
    t_plate_init = time.perf_counter() - t0

    t0 = time.perf_counter()
    car_detector = CarDetection(
        model_path="models/yolo11n.pt",
        plate_detector=None,
        confidence_threshold=CONF,
    )
    t_car_init = time.perf_counter() - t0

    print(f"INIT  car_model={t_car_init*1000:.1f} ms | plate_yolo={t_plate_init*1000:.1f} ms")
    print(
        f"Video {INPUT_VIDEO} | {width}x{height} @ {fps:.1f} fps | "
        f"frames={total_frames or '?'} -> {OUTPUT_VIDEO}"
    )
    print("OCR enabled | retry every 10 frames until valid UK plate text")
    print("-" * 70)

    plate_link_history = defaultdict(list)
    sum_car = sum_plate = sum_ocr = sum_draw = sum_total = 0.0
    n_frames = 0
    frame_idx = 0

    writer = VideoWriterContext(OUTPUT_VIDEO, fps, (width, height))
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_frame = time.perf_counter()

            t0 = time.perf_counter()
            car_metadata, frame_size = detect_cars_only(car_detector, frame)
            t_car = time.perf_counter() - t0

            plate_detector.enable_ocr = False
            t0 = time.perf_counter()
            plates = plate_detector.track_plates_for_frame(
                frame, car_metadata, frame_idx, frame_size
            )
            t_plate = time.perf_counter() - t0

            plate_detector.enable_ocr = True
            attempts_before = {
                pid: pt.get("ocr_attempts", 0)
                for pid, pt in plate_detector.plate_tracks.items()
            }
            t0 = time.perf_counter()
            plate_detector._run_ocr_for_active_plates(frame, frame_idx, frame_size, plates)
            t_ocr = time.perf_counter() - t0

            ocr_ran_ids = []
            ocr_done_texts = []
            for pid, pt in plate_detector.plate_tracks.items():
                if pt.get("ocr_attempts", 0) > attempts_before.get(pid, 0):
                    ocr_ran_ids.append(pid)
                if pt.get("ocr_done") and pt.get("text"):
                    ocr_done_texts.append(f"{pt['text']}(p{pid})")

            t0 = time.perf_counter()
            frame_vis = frame.copy()
            s = plate_detector.last_frame_stats
            projected_set = set(s["projected_car_ids"])

            loc_state = "MODEL ON" if s["model_batch_calls"] > 0 else "MODEL OFF"
            ocr_state = f"OCR RUN n={len(ocr_ran_ids)}" if ocr_ran_ids else "OCR SKIP"
            banner_bg = (0, 140, 0) if s["model_batch_calls"] == 0 else (160, 0, 160)
            banner_lines = [
                f"Frame {frame_idx}: plate {loc_state} | {ocr_state}",
                f"car={t_car*1000:.0f}ms plate={t_plate*1000:.0f}ms ocr={t_ocr*1000:.0f}ms",
                f"texts: {', '.join(ocr_done_texts) if ocr_done_texts else '-'}",
            ]
            draw_banner(frame_vis, banner_lines, banner_bg)

            cv.putText(
                frame_vis,
                "YELLOW=car | GREEN=PROJECT | MAGENTA=MODEL | plate text=OCR result",
                (10, frame_vis.shape[0] - 15),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )

            for track_id, meta in car_metadata.items():
                x1, y1, x2, y2 = [int(v) for v in meta["bbox"]]
                cv.rectangle(frame_vis, (x1, y1), (x2, y2), COLOR_CAR, 2)
                cv.putText(
                    frame_vis,
                    f"Car:{track_id}",
                    (x1, max(70, y1 - 8)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    COLOR_CAR,
                    2,
                    cv.LINE_AA,
                )

                plate_bbox = plates.get(track_id)
                plate_id = plate_detector.car_to_plate.get(track_id)
                if plate_bbox is None or plate_id is None:
                    continue

                pt = plate_detector.plate_tracks.get(plate_id, {})
                plate_text = pt.get("text")
                ocr_attempts = pt.get("ocr_attempts", 0)
                loc_conf = pt.get("confidence")
                loc_conf_txt = f"{loc_conf:.2f}" if loc_conf is not None else "-"

                px1, py1, px2, py2 = [int(v) for v in plate_bbox]
                plate_link_history[plate_id].append(
                    (frame_idx, track_id, plate_text, ocr_attempts)
                )

                if track_id in projected_set:
                    mode, color = "PROJECT", COLOR_PROJECT
                else:
                    mode, color = "MODEL", COLOR_MODEL

                cv.rectangle(frame_vis, (px1, py1), (px2, py2), color, 3)

                if plate_text:
                    label = f"{plate_text} | loc={loc_conf_txt} | Car:{track_id}"
                    label_color = COLOR_OCR_OK
                else:
                    label = (
                        f"{mode} loc={loc_conf_txt} "
                        f"Plate:{plate_id}->Car:{track_id} att={ocr_attempts}"
                    )
                    label_color = COLOR_OCR_WAIT if ocr_attempts > 0 else color

                cv.putText(
                    frame_vis,
                    label,
                    (px1, max(70, py1 - 6)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    label_color,
                    2,
                    cv.LINE_AA,
                )

                if SAVE_FRAME_IMAGES:
                    pad = 3
                    h, w = frame.shape[:2]
                    cx1, cy1 = max(0, px1 - pad), max(0, py1 - pad)
                    cx2, cy2 = min(w, px2 + pad), min(h, py2 + pad)
                    plate_crop = frame[cy1:cy2, cx1:cx2].copy()
                    if plate_crop.size > 0:
                        tag = plate_text if plate_text else f"{mode}|att{ocr_attempts}"
                        cv.putText(
                            plate_crop,
                            tag,
                            (5, 20),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            label_color,
                            2,
                            cv.LINE_AA,
                        )
                        suffix = plate_text if plate_text else mode
                        cv.imwrite(
                            os.path.join(
                                OUTPUT_DIR,
                                f"frame{frame_idx:04d}_car{track_id}_plate{plate_id}_{suffix}.jpg",
                            ),
                            plate_crop,
                        )

            if SAVE_FRAME_IMAGES:
                cv.imwrite(
                    os.path.join(OUTPUT_DIR, f"frame{frame_idx:04d}_annotated.jpg"),
                    frame_vis,
                )
            writer.write(frame_vis)
            t_draw = time.perf_counter() - t0
            t_total = time.perf_counter() - t_frame

            n_frames += 1
            sum_car += t_car
            sum_plate += t_plate
            sum_ocr += t_ocr
            sum_draw += t_draw
            sum_total += t_total

            if n_frames == 1 or n_frames % 30 == 0:
                print(
                    f"Frame {frame_idx:04d} | cars={len(car_metadata)} plates={len(plates)} | "
                    f"1.car={t_car*1000:6.1f}ms | "
                    f"2.plate={t_plate*1000:6.1f}ms | "
                    f"3.ocr={t_ocr*1000:6.1f}ms (ran={len(ocr_ran_ids)}) | "
                    f"4.draw={t_draw*1000:6.1f}ms | "
                    f"TOTAL={t_total*1000:6.1f}ms | "
                    f"texts={ocr_done_texts or '-'}"
                )

            if SHOW_WINDOWS:
                cv.imshow("cars + plates + OCR", frame_vis)
                key = cv.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break

            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        if SHOW_WINDOWS:
            cv.destroyAllWindows()

    if n_frames:
        print("-" * 70)
        print(
            f"AVG over {n_frames} frames | "
            f"car={sum_car/n_frames*1000:.1f}ms | "
            f"plate={sum_plate/n_frames*1000:.1f}ms | "
            f"ocr={sum_ocr/n_frames*1000:.1f}ms | "
            f"draw={sum_draw/n_frames*1000:.1f}ms | "
            f"total={sum_total/n_frames*1000:.1f}ms"
        )

    print("-" * 70)
    print("OCR tracks:")
    for pid, pt in sorted(plate_detector.plate_tracks.items()):
        print(
            f"  Plate {pid} -> Car {pt.get('vehicle_id')} | "
            f"text={pt.get('text')} | done={pt.get('ocr_done')} | "
            f"attempts={pt.get('ocr_attempts')}"
        )
    print(f"Exported video: {OUTPUT_VIDEO}")
    print(f"Debug dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
