import cv2 as cv
import pickle
import json
import os
import queue
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from ultralytics import YOLO

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


class CarDetection:
    CLASSIFIABLE_YOLO_CLASSES = {"car", "truck"}

    _DRAW_COLORS = {
        "id": (255, 255, 255),
        "type": (255, 200, 100),
        "mmr": (100, 255, 255),
        "color": (150, 255, 150),
    }

    def __init__(
        self,
        model_path,
        type_classifier=None,
        color_detector=None,
        make_model_detector=None,
        plate_detector=None,
        confidence_threshold=0.6,
        plate_roi_ratio=0.45,
    ):
        self.model = YOLO(model_path)
        self.type_classifier = type_classifier
        self.color_detector = color_detector
        self.make_model_detector = make_model_detector
        self.plate_detector = plate_detector
        self.confidence_threshold = confidence_threshold
        self.plate_roi_ratio = plate_roi_ratio

        self.tracking_log = {}
        self._log_lock = threading.Lock()

        # ── Producer-Consumer Queue Architecture ──
        self._analysis_queue = queue.Queue(maxsize=30)
        self._result_queue = queue.Queue(maxsize=30)
        self._stop_event = threading.Event()
        self._worker_thread = None
        self._writer_thread = None

        # ── ThreadPool for parallel analysis (color / type / mmr) ──
        self._executor = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="Analyzer"
        )

        # ── Thread-safety lock for OCR reader (defensive) ──
        self._ocr_lock = threading.Lock()

        # ── Profiling / Instrumentation (thread-safe) ──
        self._stats_lock = threading.Lock()
        self._stats = defaultdict(list)            # stage_name -> [durations بالثانية]
        self._thread_names = defaultdict(set)        # stage_name -> {أسماء الـ threads يلي نفذوها فعليًا}
        self._queue_depth_samples = {
            "analysis_queue": deque(maxlen=2000),
            "result_queue": deque(maxlen=2000),
        }

    # =========================================================
    # PUBLIC API: Streaming Entry Point
    # =========================================================
    def process_streaming(
        self,
        input_video_path,
        output_video_path,
        read_from_stub=False,
        stub_path=None,
    ):
        """
        المعالجة الستريمينج الكاملة: 3 threads بتشتغل بالتوازي الحقيقي (overlap):
          Main Thread     (Producer) : قراءة الفريم + tracking تسلسلي فقط، ما بستنى نتيجة.
          AnalysisConsumer            : OCR Gate + ThreadPoolExecutor (color/type/mmr) + رسم + لوج.
          ResultWriter                : كتابة الفريم المرسوم للفيديو + تجميع النتائج.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                cached = pickle.load(f)
            self._playback_cached(output_video_path, cached)
            return

        cap = cv.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_video_path}")

        fps = cap.get(cv.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(output_video_path, fourcc, fps, (w, h))

        all_car_detections = []
        # الـ writer + all_car_detections بيتسلموا لـ start() عشان الـ Writer Thread
        # يستهلكهم لحاله، بدون ما الـ Main Thread يلمسهم بعد هيك.
        self.start(writer=writer, all_car_detections=all_car_detections)

        frame_idx = 0
        pipeline_t0 = time.perf_counter()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # ── PRODUCER: Tracking تسلسلي فقط (إجباري بسبب persist=True) ──
                with self._timed("tracking"):
                    results = self.model.track(
                        frame,
                        persist=True,
                        iou=0.1,
                        conf=self.confidence_threshold,
                        verbose=False,
                    )[0]

                    car_metadata = self._extract_metadata(frame, results)

                    # نسخ الـ crops لتفادي مشاكل الـ memory views
                    for tid in car_metadata:
                        car_metadata[tid]["crop"] = car_metadata[tid]["crop"].copy()

                # ── put فقط، بدون أي get بعده — هون بالضبط منكسب الـ overlap الحقيقي ──
                self._analysis_queue.put(
                    (frame.copy(), frame_idx, car_metadata, (w, h))
                )
                self._sample_queue_depth()
                frame_idx += 1

        finally:
            cap.release()
            # poison pill لـ analysis_queue: الـ consumer بيفضّي الباقي المتراكم أولاً
            # وبعدين يمرر poison pill لـ result_queue بنفسه، فالـ writer بياخد كل شي
            # قبل ما يقفل ويسوي writer.release() لحاله. ولا فريم بينضاع.
            self.stop()

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(all_car_detections, f)

        self.save_tracking_log("tracking_log.json")
        total_time = time.perf_counter() - pipeline_t0
        fps_overall = frame_idx / total_time if total_time > 0 else 0.0
        print(
            f"[DONE] Processed {frame_idx} frames in {total_time:.2f}s "
            f"({fps_overall:.2f} FPS overall). Log saved."
        )
        self.print_performance_report()

    # =========================================================
    # LIFECYCLE: Start / Stop (Clean Shutdown)
    # =========================================================
    def start(self, writer=None, all_car_detections=None):
        """
        تشغيل الـ Consumer Thread + الـ Writer Thread.
        writer/all_car_detections اختياريين — لو ما انعطوا، بتضل قادر تسحب
        من self._result_queue يدويًا بدل ما يكون في Writer يفضّيها أوتوماتيك.
        """
        self._stop_event.clear()

        self._worker_thread = threading.Thread(
            target=self._consumer_loop,
            name="AnalysisConsumer",
            # daemon=True: شبكة أمان — لو الـ main thread طاح/خرج بشكل غير طبيعي
            # قبل ما ينادي stop()، هاد الـ thread ما بيمنع خروج البروسيس ويصير "معلّق حي".
            # الإغلاق النظيف (بدون فقدان فريمات) لسا مضمون عبر drain في stop().
            daemon=True,
        )
        self._worker_thread.start()

        self._writer_thread = None
        if writer is not None:
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                args=(writer, all_car_detections),
                name="ResultWriter",
                daemon=True,
            )
            self._writer_thread.start()

    def stop(self, timeout=30.0):
        """
        إيقاف مرتّب وآمن (drain-then-stop، مش stop-then-drop):
        1. poison pill بـ put() الحاجزة (مش put_nowait) — لازم توصل مهما استنّت،
           عشان ما تنضاع الإشارة لو الطابور مليان.
        2. الـ consumer بيفضّي كل الفريمات المتراكمة قبل ما ياخد الـ poison pill،
           وبعدين هو نفسه بيمرر poison pill لـ result_queue.
        3. ننتظر الـ Writer يخلص يكتب كل شي متراكم ويقفل الفيديو.
        4. نقفل الـ ThreadPoolExecutor.
        النتيجة: ولا فريم بينضاع، وولا thread بيضل حي بعد الرجوع من stop().
        """
        self._analysis_queue.put(None)  # poison pill — بلوكينج، لازم توصل

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)
            if self._worker_thread.is_alive():
                print(
                    "[WARN] AnalysisConsumer لسا شغال بعد "
                    f"{timeout}s — فيه bottleneck أو تعليقة بمكوّن من المكونات."
                )

        if self._writer_thread is not None:
            self._writer_thread.join(timeout=timeout)
            if self._writer_thread.is_alive():
                print(f"[WARN] ResultWriter لسا شغال بعد {timeout}s — تحقق من كتابة الفيديو.")

        self._executor.shutdown(wait=True, cancel_futures=True)
        self._stop_event.set()  # علم حالة "مقفول" — ما بيتحكم بأي loop، بس للاستعلام

    def is_running(self):
        return not self._stop_event.is_set()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass  # ما بدنا استثناء بـ __del__ يطيح البروسيس وقت التنظيف

    # =========================================================
    # CONSUMER LOOP (Single Worker)
    # =========================================================
    def _consumer_loop(self):
        """
        مستهلك وحيد (FIFO). بيضل شغال بـ blocking get() لحد ما ياخد poison pill
        (None) بالذات — مش لحد ما ينحط stop_event. هيك أي فريمات متراكمة بالطابور
        وقت نداء stop() بتتفضّى بالكامل قبل ما الـ thread يخلص (drain-then-stop).

        1. OCR Logic Gate (Plate Detection + OCR)
        2. Filtered Batch → ThreadPoolExecutor (Color / Type / MMR)
        3. Collect → Draw → Log → إرسال لـ Writer Thread
        """
        while True:
            item = self._analysis_queue.get()  # blocking — بدون polling/timeout

            if item is None:  # poison pill من stop()
                break

            self._sample_queue_depth()
            frame, frame_idx, car_metadata, frame_size = item
            frame_t0 = time.perf_counter()

            try:
                # ── المرحلة 1: OCR Logic Gate (تسلسلي، قبل أي تفريع) ──
                with self._timed("ocr_gate"):
                    plates_data, filtered_metadata = self._apply_ocr_gate(
                        frame, car_metadata, frame_idx, frame_size
                    )

                # ── المرحلة 2: Parallel Analysis على الـ Filtered Batch فقط ──
                with self._timed("parallel_stage_wall"):
                    if filtered_metadata:
                        colors, types, mmrs = self._run_parallel_analysis(
                            frame, filtered_metadata, frame_idx
                        )
                    else:
                        colors, types, mmrs = {}, {}, {}

                # ── المرحلة 3: بناء النتائج النهائية ──
                car_list = self._build_final_results(
                    car_metadata, filtered_metadata, plates_data, colors, types, mmrs
                )

                # ── المرحلة 4: الرسم والتسجيل ──
                with self._timed("draw"):
                    self.draw_frame(frame, car_list)
                with self._timed("log"):
                    for car in car_list:
                        self._log_from_dict(frame_idx, car)

                # ── المرحلة 5: إرسال للـ Writer Thread (بدون انتظار منه) ──
                self._result_queue.put((frame_idx, frame, car_list))
                self._sample_queue_depth()

                # ── المرحلة 6: تنظيف المسارات الميتة ──
                self._cleanup_detectors(frame_idx)

                with self._stats_lock:
                    self._stats["frame_total"].append(time.perf_counter() - frame_t0)

            except Exception as exc:
                # فريم وحد فاشل ما لازم يطيح الـ Consumer كامل (يعني ما رح يعلّق
                # الـ Producer على طابور مليان للأبد لو صار خطأ غير متوقع بمكوّن).
                print(f"[ERROR] Consumer failed on frame {frame_idx}: {exc!r}")

        # خلصنا نفضّي analysis_queue بالكامل → مرر poison pill لـ result_queue
        self._result_queue.put(None)

    def _writer_loop(self, writer, all_car_detections):
        """
        Consumer ثاني بيسحب من result_queue بالترتيب (FIFO) ويكتب الفيديو —
        بدون ما يخلي الـ Producer (main thread) ينتظره. هون بالضبط منكسب الـ
        overlap الحقيقي: بينما هاد الـ thread عم يكتب فريم N-1، الـ Consumer عم
        يحلل فريم N، والـ Producer عم يتتبع فريم N+1 — الثلاثة بنفس اللحظة.
        """
        while True:
            item = self._result_queue.get()
            if item is None:  # poison pill من الـ consumer
                break

            frame_idx, drawn_frame, car_list = item
            with self._timed("write"):
                writer.write(drawn_frame)
            if all_car_detections is not None:
                all_car_detections.append(car_list)

        writer.release()

    # =========================================================
    # OCR LOGIC GATE (Single Responsibility)
    # =========================================================
    def _apply_ocr_gate(self, frame, car_metadata, frame_idx, frame_size):
        """
        تشغيل Plate Detection & OCR لكل السيارات في الفريم.
        ترجع:
            plates_data: {track_id: plate_bbox}
            filtered_metadata: السيارات التي اجتازت شرط OCR فقط.
        """
        if self.plate_detector is None or not car_metadata:
            return {}, {}

        # Thread-safety: lock حول OCR reader إذا لزم (defensive)
        with self._ocr_lock:
            plates_data = self.plate_detector.track_plates_for_frame(
                frame, car_metadata, frame_idx, frame_size
            )

        filtered_metadata = {}
        for track_id, meta in car_metadata.items():
            plate_track_id = self.plate_detector.car_to_plate.get(track_id)
            if plate_track_id is None:
                continue

            pt = self.plate_detector.plate_tracks.get(plate_track_id, {})
            text = pt.get("text")

            # شرط الـ Gate: يجب أن يكون النص صالحاً (ليس None ولا فارغ)
            if text and str(text).strip():
                filtered_metadata[track_id] = meta

        return plates_data, filtered_metadata

    # =========================================================
    # PARALLEL ANALYSIS (ThreadPoolExecutor)
    # =========================================================
    def _run_parallel_analysis(self, frame, filtered_metadata, frame_idx):
        """
        تشغيل Color + Type + MMR بالتوازي عبر ThreadPoolExecutor.
        الـ Consumer ينتظر حتى تكتمل الثلاثة.
        """
        car_crops = {
            tid: filtered_metadata[tid]["bbox"] for tid in filtered_metadata
        }
        cls_name_dict = {
            tid: filtered_metadata[tid]["cls_name"] for tid in filtered_metadata
        }

        futures = {}

        if self.color_detector is not None:
            futures["colors"] = self._executor.submit(
                self._timed_call,
                "color",
                self.color_detector.get_stable_colors_for_frame,
                frame,
                car_crops,
                frame_idx,
            )

        if self.type_classifier is not None:
            futures["types"] = self._executor.submit(
                self._timed_call,
                "type",
                self.type_classifier.classify_and_vote_for_frame,
                frame,
                car_crops,
                cls_name_dict,
                frame_idx,
            )

        if self.make_model_detector is not None:
            futures["mmrs"] = self._executor.submit(
                self._timed_call,
                "mmr",
                self.make_model_detector.get_stable_make_models_for_frame,
                frame,
                car_crops,
                frame_idx,
            )

        colors = futures["colors"].result() if "colors" in futures else {}
        types = futures["types"].result() if "types" in futures else {}
        mmrs = futures["mmrs"].result() if "mmrs" in futures else {}

        return colors, types, mmrs

    # =========================================================
    # RESULT BUILDER
    # =========================================================
    def _build_final_results(
        self, car_metadata, filtered_metadata, plates_data, colors, types, mmrs
    ):
        """
        تجميع قائمة النتائج النهائية.
        السيارات المرفوضة (بدون OCR) تُدرج ببيانات YOLO فقط.
        """
        car_list = []
        for track_id, meta in car_metadata.items():
            passed_gate = track_id in filtered_metadata

            color_name, color_vote = (
                colors.get(track_id, ("Unknown", 0.0))
                if passed_gate
                else ("Unknown", 0.0)
            )
            final_type, final_conf = (
                types.get(track_id, ("Unknown", 0.0))
                if passed_gate
                else ("Unknown", 0.0)
            )
            make_model_name, make_model_conf = (
                mmrs.get(track_id, ("Unknown", 0.0))
                if passed_gate
                else ("Unknown", 0.0)
            )

            plate_bbox = plates_data.get(track_id, None)
            plate_track_id = None
            plate_text = None
            plate_text_conf = None

            if self.plate_detector is not None:
                plate_track_id = self.plate_detector.car_to_plate.get(track_id)
                if plate_track_id is not None:
                    pt = self.plate_detector.plate_tracks.get(plate_track_id, {})
                    plate_text = pt.get("text")
                    plate_text_conf = pt.get("text_conf")

            car_list.append(
                {
                    "bbox": meta["bbox"],
                    "track_id": track_id,
                    "yolo_class": meta["cls_name"],
                    "yolo_conf": meta["yolo_conf"],
                    "type_name": final_type,
                    "type_conf": final_conf,
                    "color_name": color_name,
                    "color_conf": color_vote,
                    "make_model_name": make_model_name,
                    "make_model_conf": make_model_conf,
                    "plate_bbox": plate_bbox,
                    "plate_track_id": plate_track_id,
                    "plate_text": plate_text,
                    "plate_text_conf": plate_text_conf,
                }
            )
        return car_list

    # =========================================================
    # PRODUCER HELPERS
    # =========================================================
    def _extract_metadata(self, frame, results):
        """استخراج بيانات السيارات من نتيجة YOLO track."""
        id_name_dict = results.names
        car_metadata = {}
        for box in results.boxes:
            bbox = box.xyxy.tolist()[0]
            cls_id = int(box.cls.tolist()[0])
            cls_name = id_name_dict[cls_id]
            track_id = int(box.id.item()) if box.id is not None else -1
            yolo_conf = float(box.conf.item()) if box.conf is not None else 0.0

            if cls_name not in VEHICLE_CLASSES:
                continue
            if track_id == -1:
                continue

            crop = self._crop(frame, bbox)
            if crop is None or crop.size == 0:
                continue

            car_metadata[track_id] = {
                "bbox": bbox,
                "cls_name": cls_name,
                "yolo_conf": yolo_conf,
                "crop": crop,
            }
        return car_metadata

    def _playback_cached(self, output_video_path, cached_detections):
        """إعادة رسم من stub مخزن (بدون إعادة كشف)."""
        # يمكن تنفيذها لاحقاً حسب الحاجة
        pass

    # =========================================================
    # PROFILING / INSTRUMENTATION
    # =========================================================
    @contextmanager
    def _timed(self, stage_name):
        """قياس زمن أي مرحلة + اسم الـ thread يلي نفذتها فعليًا."""
        t0 = time.perf_counter()
        yield
        dt = time.perf_counter() - t0
        with self._stats_lock:
            self._stats[stage_name].append(dt)
            self._thread_names[stage_name].add(threading.current_thread().name)

    def _timed_call(self, stage_name, func, *args, **kwargs):
        """
        نفس فكرة _timed بس كدالة قابلة للتمرير لـ executor.submit مباشرة —
        لازم القياس ياخذ مكان جوا thread الـ pool نفسه، مو جوا thread الاستدعاء،
        وإلا رح نقيس زمن التسليم مش زمن التنفيذ الفعلي، ولا رح نعرف مين نفذ شو.
        """
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        with self._stats_lock:
            self._stats[stage_name].append(dt)
            self._thread_names[stage_name].add(threading.current_thread().name)
        return result

    def _sample_queue_depth(self):
        """
        أهم مؤشر على إذا في overlap حقيقي أو لأ: لو الطابورين قريبين من صفر
        دايمًا، معناتها ما في backlog وما في pipelining فعلي (رجعنا لنفس مشكلة
        الـ synchronous rendezvous القديمة). لو في تذبذب حقيقي بالعمق، هاد دليل
        إنه الـ 3 threads (Producer/Consumer/Writer) شغالين بالتوازي فعليًا.
        """
        with self._stats_lock:
            self._queue_depth_samples["analysis_queue"].append(self._analysis_queue.qsize())
            self._queue_depth_samples["result_queue"].append(self._result_queue.qsize())

    def report_active_threads(self):
        """قائمة كل الـ threads الحية حاليًا بالبروسيس — للتأكد ما في dead/leaked threads."""
        return [t.name for t in threading.enumerate()]

    def get_performance_report(self):
        """
        ملخص أداء كل مرحلة: عدد المرات، متوسط/أدنى/أعلى زمن، FPS مكافئ، وأسماء
        الـ threads يلي فعليًا نفذت هالمرحلة. بالإضافة لـ parallel_efficiency_ratio:
        (متوسط زمن color + متوسط زمن type + متوسط زمن mmr) / متوسط زمن parallel_stage_wall
          - قريب من 3  → تسريع حقيقي ممتاز (المكونات فعلاً بتشتغل بالتوازي).
          - قريب من 1  → ولا تسريع، تنفيذ متسلسل فعليًا رغم استخدام threads
            (الاحتمال الأكبر: GIL contention لأنه المكونات Python-bound وما
            بتحرر الـ GIL كفاية أثناء التنفيذ).
        """
        with self._stats_lock:
            report = {}
            for stage, durations in self._stats.items():
                if not durations:
                    continue
                n = len(durations)
                total = sum(durations)
                report[stage] = {
                    "count": n,
                    "avg_ms": round(total / n * 1000, 2),
                    "min_ms": round(min(durations) * 1000, 2),
                    "max_ms": round(max(durations) * 1000, 2),
                    "equivalent_fps": round(n / total, 2) if total > 0 else None,
                    "threads_used": sorted(self._thread_names.get(stage, [])),
                }

            depths = {
                name: (
                    round(sum(samples) / len(samples), 2) if samples else 0,
                    max(samples) if samples else 0,
                )
                for name, samples in self._queue_depth_samples.items()
            }

        parallel_efficiency = None
        component_stages = [s for s in ("color", "type", "mmr") if s in report]
        if component_stages and "parallel_stage_wall" in report:
            sum_components_avg = sum(report[s]["avg_ms"] for s in component_stages)
            wall_avg = report["parallel_stage_wall"]["avg_ms"]
            if wall_avg > 0:
                parallel_efficiency = round(sum_components_avg / wall_avg, 2)

        return {
            "stages": report,
            "queue_depth_avg_max": depths,
            "parallel_efficiency_ratio": parallel_efficiency,
            "active_threads_now": self.report_active_threads(),
        }

    def print_performance_report(self):
        report = self.get_performance_report()
        print("\n" + "=" * 70)
        print("PERFORMANCE REPORT")
        print("=" * 70)
        for stage, s in report["stages"].items():
            print(
                f"{stage:20s} | n={s['count']:5d} | avg={s['avg_ms']:7.2f}ms | "
                f"min={s['min_ms']:7.2f}ms | max={s['max_ms']:7.2f}ms | "
                f"~{s['equivalent_fps']:6.1f} fps | threads={s['threads_used']}"
            )
        print("-" * 70)
        for q, (avg_depth, max_depth) in report["queue_depth_avg_max"].items():
            print(f"{q:20s} | avg_depth={avg_depth} | max_depth={max_depth}")
        if report["parallel_efficiency_ratio"] is not None:
            print("-" * 70)
            print( "Parallel efficiency (sum(color + type + mmr) / parallel_wall) = " \
            "" f"{report['parallel_efficiency_ratio']} " "(close to 3 = excellent real speedup | close to 1 = no speedup, likely GIL bottleneck)" )
        print("-" * 70)
        print(f"Active threads now: {report['active_threads_now']}")
        print("=" * 70 + "\n")

    # =========================================================
    # CLEANUP
    # =========================================================
    def _cleanup_detectors(self, frame_idx):
        if self.type_classifier is not None:
            self.type_classifier.cleanup_inactive_tracks(frame_idx)
        if self.color_detector is not None:
            self.color_detector.cleanup_inactive_tracks(frame_idx)
        if self.make_model_detector is not None:
            self.make_model_detector.cleanup_inactive_tracks(frame_idx)

    # =========================================================
    # CROP & CLIP (DRY)
    # =========================================================
    def _crop(self, frame, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _clip_bbox(self, frame, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return x1, y1, x2, y2

    # =========================================================
    # LOGGING
    # =========================================================
    def _log_from_dict(self, frame_idx, detection):
        self._log_prediction(
            track_id=detection["track_id"],
            frame_idx=frame_idx,
            yolo_class=detection["yolo_class"],
            yolo_conf=detection["yolo_conf"],
            smoothed_type=detection["type_name"],
            type_conf=detection["type_conf"],
            color_name=detection["color_name"],
            color_vote=detection["color_conf"],
            make_model_name=detection["make_model_name"],
            make_model_conf=detection["make_model_conf"],
            plate_bbox=detection.get("plate_bbox"),
            plate_track_id=detection.get("plate_track_id"),
            plate_text=detection.get("plate_text"),
            plate_text_conf=detection.get("plate_text_conf"),
        )

    def _log_prediction(
        self,
        track_id,
        frame_idx,
        yolo_class,
        yolo_conf,
        smoothed_type,
        type_conf,
        color_name,
        color_vote,
        make_model_name,
        make_model_conf,
        plate_bbox=None,
        plate_track_id=None,
        plate_text=None,
        plate_text_conf=None,
    ):
        with self._log_lock:
            if track_id not in self.tracking_log:
                self.tracking_log[track_id] = []

            self.tracking_log[track_id].append(
                {
                    "frame": frame_idx,
                    "yolo_class": yolo_class,
                    "yolo_confidence": round(yolo_conf, 3) if yolo_conf is not None else 0.0,
                    "predicted_type": smoothed_type,
                    "type_confidence": round(type_conf, 3) if type_conf is not None else 0.0,
                    "smoothed_color": color_name,
                    "color_vote_ratio": round(color_vote, 3) if color_vote is not None else 0.0,
                    "make_model": make_model_name,
                    "make_model_confidence": round(make_model_conf, 3)
                    if make_model_conf is not None
                    else 0.0,
                    "plate_bbox": plate_bbox,
                    "plate_track_id": plate_track_id,
                    "plate_text": plate_text,
                    "plate_text_conf": round(plate_text_conf, 3)
                    if plate_text_conf is not None
                    else None,
                }
            )

    def save_tracking_log(self, output_path="tracking_log.json"):
        with self._log_lock:
            snapshot = {tid: list(entries) for tid, entries in self.tracking_log.items()}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(snapshot)} vehicles to: {output_path}")

    # =========================================================
    # DRAWING (unchanged logic, isolated)
    # =========================================================
    @staticmethod
    def _draw_label(frame, text, x, y, bg_color, font_scale=0.45, thickness=1):
        (text_w, text_h), _ = cv.getTextSize(
            text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        y = max(y, text_h + 6)
        pad = 4
        cv.rectangle(
            frame,
            (x, y - text_h - pad),
            (x + text_w + pad * 2, y + pad),
            bg_color,
            -1,
        )
        cv.putText(
            frame,
            text,
            (x + pad, y - 2),
            cv.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 1,
            cv.LINE_AA,
        )
        cv.putText(
            frame,
            text,
            (x + pad, y - 2),
            cv.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv.LINE_AA,
        )

    def draw_frame(self, frame, car_list):
        COLORS = self._DRAW_COLORS
        for detection in car_list:
            x1, y1, x2, y2 = detection["bbox"]
            track_id = detection["track_id"]
            lines = []

            id_prefix = f"ID:{track_id} | " if track_id != -1 else ""
            lines.append(
                (
                    f"{id_prefix}{detection['yolo_class']} ({detection['yolo_conf']:.2f})",
                    COLORS["id"],
                )
            )
            lines.append(
                (f"Type: {detection['type_name']} ({detection['type_conf']:.2f})", COLORS["type"])
            )
            if detection.get("make_model_name") and detection["make_model_name"] != "Unknown":
                lines.append(
                    (
                        f"MMR: {detection['make_model_name']} ({detection['make_model_conf']:.2f})",
                        COLORS["mmr"],
                    )
                )
            lines.append(
                (f"Color: {detection['color_name']} ({detection['color_conf']:.2f})", COLORS["color"])
            )

            line_height = 18
            total_height = len(lines) * line_height + 4
            start_y = int(y1) - total_height
            if start_y < 10:
                start_y = int(y2) + 18

            current_y = start_y
            for text, color in lines:
                self._draw_label(frame, text, int(x1), current_y, color)
                current_y += line_height

            cv.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 255),
                2,
            )

            if detection.get("plate_bbox") is not None:
                px1, py1, px2, py2 = [int(v) for v in detection["plate_bbox"]]
                cv.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                plate_text = detection.get("plate_text")
                if plate_text:
                    plate_label = str(plate_text)
                else:
                    plate_id = detection.get("plate_track_id")
                    plate_label = f"Plate:{plate_id}" if plate_id is not None else "Plate"
                self._draw_label(frame, plate_label, px1, py1 - 4, (0, 0, 255))

        return frame

    def draw_bboxes(self, video_frames, car_detections):
        output_frames = []
        for frame, car_list in zip(video_frames, car_detections):
            self.draw_frame(frame, car_list)
            output_frames.append(frame)
        return output_frames

    # =========================================================
    # LEGACY BATCH API (محفوظ للتوافق)
    # =========================================================
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        car_detections = [
            self.detect_frame(frame, idx) for idx, frame in enumerate(frames)
        ]

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(car_detections, f)
        return car_detections

    def detect_frame(self, frame, frame_idx):
        """
        النسخة القديمة (batch per frame) — محفوظة للتوافق.
        يُفضل استخدام process_streaming() للأداء.
        """
        frame_height, frame_width = frame.shape[:2]
        results = self.model.track(
            frame, persist=True, iou=0.1, conf=self.confidence_threshold, verbose=False
        )[0]
        id_name_dict = results.names
        car_list = []

        car_crops = {}
        car_metadata = {}

        for box in results.boxes:
            bbox = box.xyxy.tolist()[0]
            cls_id = int(box.cls.tolist()[0])
            cls_name = id_name_dict[cls_id]
            track_id = int(box.id.item()) if box.id is not None else -1
            yolo_conf = float(box.conf.item()) if box.conf is not None else 0.0

            if cls_name not in VEHICLE_CLASSES:
                continue

            crop = self._crop(frame, bbox)
            if track_id != -1 and crop is not None:
                car_crops[track_id] = bbox
                car_metadata[track_id] = {
                    "bbox": bbox,
                    "cls_name": cls_name,
                    "yolo_conf": yolo_conf,
                    "crop": crop,
                }

        plates = {}
        if self.plate_detector is not None and car_crops:
            plates = self.plate_detector.track_plates_for_frame(
                frame, car_metadata, frame_idx, (frame_width, frame_height)
            )

        colors = {}
        if self.color_detector is not None and car_crops:
            colors = self.color_detector.get_stable_colors_for_frame(
                frame, car_crops, frame_idx
            )

        types = {}
        if self.type_classifier is not None and car_crops:
            cls_name_dict = {tid: car_metadata[tid]["cls_name"] for tid in car_crops}
            types = self.type_classifier.classify_and_vote_for_frame(
                frame, car_crops, cls_name_dict, frame_idx
            )

        mmrs = {}
        if self.make_model_detector is not None and car_crops:
            mmrs = self.make_model_detector.get_stable_make_models_for_frame(
                frame, car_crops, frame_idx
            )

        for track_id, meta in car_metadata.items():
            color_name, color_vote = colors.get(track_id, ("Unknown", 0.0))
            final_type, final_conf = types.get(track_id, ("Unknown", 0.0))
            make_model_name, make_model_conf = mmrs.get(track_id, ("Unknown", 0.0))
            plate_bbox = plates.get(track_id, None)

            plate_track_id = None
            plate_text = None
            plate_text_conf = None
            if self.plate_detector is not None:
                plate_track_id = self.plate_detector.car_to_plate.get(track_id)
                if plate_track_id is not None:
                    pt = self.plate_detector.plate_tracks.get(plate_track_id, {})
                    plate_text = pt.get("text")
                    plate_text_conf = pt.get("text_conf")

            self._log_prediction(
                track_id,
                frame_idx,
                meta["cls_name"],
                meta["yolo_conf"],
                final_type,
                final_conf,
                color_name,
                color_vote,
                make_model_name,
                make_model_conf,
                plate_bbox,
                plate_track_id,
                plate_text,
                plate_text_conf,
            )

            car_list.append(
                {
                    "bbox": meta["bbox"],
                    "track_id": track_id,
                    "yolo_class": meta["cls_name"],
                    "yolo_conf": meta["yolo_conf"],
                    "type_name": final_type,
                    "type_conf": final_conf,
                    "color_name": color_name,
                    "color_conf": color_vote,
                    "make_model_name": make_model_name,
                    "make_model_conf": make_model_conf,
                    "plate_bbox": plate_bbox,
                    "plate_track_id": plate_track_id,
                    "plate_text": plate_text,
                    "plate_text_conf": plate_text_conf,
                }
            )

        if self.type_classifier is not None:
            self.type_classifier.cleanup_inactive_tracks(frame_idx)
        if self.color_detector is not None:
            self.color_detector.cleanup_inactive_tracks(frame_idx)
        if self.make_model_detector is not None:
            self.make_model_detector.cleanup_inactive_tracks(frame_idx)

        return car_list
