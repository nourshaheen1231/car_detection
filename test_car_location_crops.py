"""
تيست: كشف السيارات + التحقق من أن التتبع (track_id) يستمر بين الفريمات.
يشغّل YOLO tracking فقط (بدون لون/نوع/MMR/لوحة).

كيف تتأكد إن التتبع شغال؟
- نفس السيارة لازم تحتفظ بنفس ID عبر الفريمات المتتالية
- شوف قسم TRACKING في الطباعة: kept / new / lost
- لو ID بيتغيّر كل فريم لنفس السيارة → التتبع مش ثابت
"""

import os
from collections import defaultdict

import cv2 as cv

from detections import CarDetection
from utils import iter_video_frames

INPUT_VIDEO = "input_videos/input_video9.mp4"
MAX_FRAMES = 5
OUTPUT_DIR = "debug_car_crops"
SHOW_WINDOWS = True
CONF = 0.55


def main():
    if not os.path.exists(INPUT_VIDEO):
        raise FileNotFoundError(f"الفيديو غير موجود: {INPUT_VIDEO}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    car_detector = CarDetection(
        model_path="models/yolo11n.pt",
        confidence_threshold=CONF,
    )

    print(f"Video: {INPUT_VIDEO}")
    print(f"Processing first {MAX_FRAMES} frames...")
    print(f"Crops: {OUTPUT_DIR}/")
    print("-" * 60)

    total_cars = 0
    prev_ids = set()
    # track_id -> list of frame indices where it appeared
    id_history = defaultdict(list)

    for frame_idx, frame in enumerate(iter_video_frames(INPUT_VIDEO)):
        if frame_idx >= MAX_FRAMES:
            break

        car_list = car_detector.detect_frame(frame, frame_idx)
        curr_ids = {det["track_id"] for det in car_list if det["track_id"] != -1}

        kept = sorted(prev_ids & curr_ids)
        new_ids = sorted(curr_ids - prev_ids)
        lost_ids = sorted(prev_ids - curr_ids)

        print(f"\n===== Frame {frame_idx} | {len(car_list)} vehicle(s) =====")
        print(f"  IDs now : {sorted(curr_ids) if curr_ids else '-'}")
        if frame_idx > 0:
            print(f"  TRACKING | kept={kept or '-'} | new={new_ids or '-'} | lost={lost_ids or '-'}")
            if kept:
                print(f"  => التتبع شغال لهذه الـ IDs (نفس السيارة استمرت): {kept}")
            elif prev_ids and curr_ids and not kept:
                print("  => تحذير: في سيارات بالفريم السابق والحالي بس ما في ID مشترك (ممكن تبديل ID)")
        else:
            print(f"  TRACKING | أول فريم — IDs جديدة: {sorted(curr_ids) or '-'}")

        frame_vis = frame.copy()

        for det in car_list:
            bbox = det["bbox"]
            track_id = det["track_id"]
            cls_name = det["yolo_class"]
            conf = det["yolo_conf"]

            if track_id != -1:
                id_history[track_id].append(frame_idx)

            crop = car_detector._crop(frame, bbox)
            if crop is None or crop.size == 0:
                print(f"  skip empty crop | id={track_id}")
                continue

            total_cars += 1
            status = ""
            if frame_idx > 0:
                if track_id in kept:
                    status = "KEPT"
                elif track_id in new_ids:
                    status = "NEW"
                else:
                    status = "?"

            out_name = f"frame{frame_idx:02d}_id{track_id}_{cls_name}_{conf:.2f}.jpg"
            cv.imwrite(os.path.join(OUTPUT_DIR, out_name), crop)

            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = (0, 255, 0) if status == "KEPT" else (0, 255, 255)
            if status == "NEW":
                color = (0, 165, 255)

            label = f"ID:{track_id} {cls_name} {conf:.2f}"
            if status:
                label = f"[{status}] {label}"

            cv.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
            cv.putText(
                frame_vis,
                label,
                (x1, max(20, y1 - 8)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv.LINE_AA,
            )

            print(
                f"  [{status or 'INIT'}] id={track_id} {cls_name} conf={conf:.2f} "
                f"box=({x1},{y1},{x2},{y2}) -> {out_name}"
            )

            if SHOW_WINDOWS:
                cv.imshow(f"crop id{track_id}", crop)

        cv.imwrite(os.path.join(OUTPUT_DIR, f"frame{frame_idx:02d}_annotated.jpg"), frame_vis)

        if SHOW_WINDOWS:
            cv.imshow("frame annotated", frame_vis)
            key = cv.waitKey(0)
            if key in (ord("q"), ord("Q"), 27):
                break

        prev_ids = curr_ids

    if SHOW_WINDOWS:
        cv.destroyAllWindows()

    print("\n" + "=" * 60)
    print("SUMMARY — ظهور كل track_id عبر الفريمات")
    print("=" * 60)
    if not id_history:
        print("ما انكشف أي مركبة.")
    else:
        multi = 0
        for tid in sorted(id_history):
            frames = id_history[tid]
            span = f"frames {frames[0]}..{frames[-1]}" if len(frames) > 1 else f"frame {frames[0]} only"
            flag = "OK tracked" if len(frames) > 1 else "seen once"
            if len(frames) > 1:
                multi += 1
            print(f"  ID {tid}: appeared {len(frames)} time(s) -> {frames} ({span}) [{flag}]")

        print("-" * 60)
        if multi > 0:
            print(f"نتيجة: {multi} معرف(ات) استمرت أكثر من فريم -> التتبع يبدو شغال.")
        else:
            print("نتيجة: ولا ID استمر لأكثر من فريم. إما السيارات بتطلع بسرعة أو التتبع ضعيف.")

    print(f"Total crops saved: {total_cars}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("ألوان البوكس: أخضر=KEPT | برتقالي=NEW | أصفر=أول فريم")


if __name__ == "__main__":
    main()
