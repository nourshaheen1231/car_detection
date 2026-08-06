import time
from collections import defaultdict


class PerformanceProfiler:
   
    def __init__(self):
        self.timings: dict[str, list] = defaultdict(list)

    def timer(self, name: str):
        return _TimerContext(self, name)

    def record(self, name: str, duration_seconds: float):
        self.timings[name].append(duration_seconds)

    def summary(self) -> dict:

        report = {}
        for name, values in self.timings.items():
            values_ms = [v * 1000 for v in values]
            report[name] = {
                "count": len(values_ms),
                "total_ms": sum(values_ms),
                "avg_ms": sum(values_ms) / len(values_ms) if values_ms else 0.0,
                "min_ms": min(values_ms) if values_ms else 0.0,
                "max_ms": max(values_ms) if values_ms else 0.0,
            }
        return report

    def fps(self, frame_timer_name: str = "frame_total") -> dict | None:

        values = self.timings.get(frame_timer_name)
        if not values:
            return None

        total_seconds = sum(values)
        count = len(values)
        avg_seconds = total_seconds / count

        return {
            "frame_count": count,
            "total_seconds": total_seconds,
            "avg_fps": 1.0 / avg_seconds if avg_seconds > 0 else float("inf"),
            "overall_fps": count / total_seconds if total_seconds > 0 else float("inf"),
            "min_fps": 1.0 / max(values) if max(values) > 0 else float("inf"),
            "max_fps": 1.0 / min(values) if min(values) > 0 else float("inf"),
        }

    def print_summary(self, title: str = "Performance Profiling Report"):
        report = self.summary()
        if not report:
            print("No measurements recorded yet.")
            return

        operation_report = {k: v for k, v in report.items() if k != "frame_total"}

        if not operation_report:
            self._print_fps_only()
            return

        grand_total = sum(stats["total_ms"] for stats in operation_report.values()) or 1.0

        col_widths = (34, 8, 14, 12, 12, 12, 10)
        header = (
            f"{'Operation':<{col_widths[0]}}"
            f"{'Count':>{col_widths[1]}}"
            f"{'Total (ms)':>{col_widths[2]}}"
            f"{'Avg (ms)':>{col_widths[3]}}"
            f"{'Min (ms)':>{col_widths[4]}}"
            f"{'Max (ms)':>{col_widths[5]}}"
            f"{'Share %':>{col_widths[6]}}"
        )
        print(f"\n{'=' * len(header)}")
        print(title)
        print(f"{'=' * len(header)}")
        print(header)
        print("-" * len(header))

        preferred_order = [
            "yolo_track",
            "color_crop_tta_build",
            "color_model_predict",
            "color_voting",
        ]
        ordered_names = [n for n in preferred_order if n in operation_report]
        ordered_names += [n for n in operation_report if n not in ordered_names]

        for name in ordered_names:
            stats = operation_report[name]
            pct = (stats["total_ms"] / grand_total) * 100
            pct_str = f"{pct:.1f}%"

            print(
                f"{name:<{col_widths[0]}}"
                f"{stats['count']:>{col_widths[1]}}"
                f"{stats['total_ms']:>{col_widths[2]}.1f}"
                f"{stats['avg_ms']:>{col_widths[3]}.2f}"
                f"{stats['min_ms']:>{col_widths[4]}.2f}"
                f"{stats['max_ms']:>{col_widths[5]}.2f}"
                f"{pct_str:>{col_widths[6]}}"
            )
        print("=" * len(header))

        fps_stats = self.fps()
        if fps_stats is not None:
            print()
            print(f"Frames processed     : {fps_stats['frame_count']}")
            print(f"Total processing time: {fps_stats['total_seconds']:.2f} s")
            print(f"Average FPS (precise): {fps_stats['avg_fps']:.2f} fps")
            print(f"Overall FPS (n/time) : {fps_stats['overall_fps']:.2f} fps")
            print(f"Slowest frame FPS    : {fps_stats['min_fps']:.2f} fps")
            print(f"Fastest frame FPS    : {fps_stats['max_fps']:.2f} fps")

    def _print_fps_only(self):
        fps_stats = self.fps()
        if fps_stats is None:
            print("No standalone operation measurements, and no frame_total timing to compute FPS.")
            return
        print(f"Frames processed     : {fps_stats['frame_count']}")
        print(f"Total processing time: {fps_stats['total_seconds']:.2f} s")
        print(f"Average FPS (precise): {fps_stats['avg_fps']:.2f} fps")
        print(f"Overall FPS (n/time) : {fps_stats['overall_fps']:.2f} fps")

    def reset(self):
        self.timings.clear()


class _TimerContext:
    __slots__ = ("profiler", "name", "_start")

    def __init__(self, profiler: PerformanceProfiler, name: str):
        self.profiler = profiler
        self.name = name

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._start
        self.profiler.record(self.name, elapsed)
        return False  