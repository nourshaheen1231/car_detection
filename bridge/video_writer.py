import shutil
import subprocess
import threading

import numpy as np


class FFmpegStreamWriter:
    """
    بديل لـ cv.VideoWriter بنفس الواجهة (write / release)،
    بس بيكتب الفريمات مباشرة بصيغة H.264 عن طريق ffmpeg pipe
    بدل ما يكتب mp4v ويحتاج تحويل منفصل بعدين.

    الاستخدام مطابق تماماً لـ cv.VideoWriter:
        writer = FFmpegStreamWriter(output_path, fps, (w, h))
        writer.write(frame)   # frame = numpy array بصيغة BGR (زي أي فريم cv2 عادي)
        writer.release()
    """

    # آخر عدد سطور من stderr نحتفظ فيهم للتشخيص عند الفشل
    _STDERR_TAIL_LINES = 40

    def __init__(self, output_path, fps, frame_size, use_gpu=False, crf=23):
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "[FFmpegStreamWriter] ffmpeg غير موجود بالـ PATH. "
                "لازم يكون منصّب ومتاح من سطر الأوامر."
            )

        w, h = frame_size
        self._expected_frame_bytes = w * h * 3  # bgr24 = 3 بايت لكل بكسل

        if use_gpu:
            # يحتاج ffmpeg مبني مع دعم NVENC + GPU متوفر فعلياً عالجهاز
            codec_args = ["-c:v", "h264_nvenc", "-preset", "p1", "-cq", str(crf)]
        else:
            codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]

        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "warning",  # يقلل الإخراج عبر stderr لأقل حد
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",          # نفس صيغة فريمات cv2 الافتراضية
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", "-",                     # قراءة الفريمات من stdin
            *codec_args,
            # libx264 مع yuv420p بيرفض أي عرض/ارتفاع فردي (odd) — هاد الفلتر
            # بيضبط الأبعاد لأقرب رقم زوجي تلقائياً، بغض النظر شو أبعاد الفيديو الأصلي
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",                         # ما في صوت بهاد المصدر (فريمات فقط)
            output_path,
        ]

        self._output_path = output_path
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._closed = False

        # مهم جداً: لازم نفرّغ stderr باستمرار بـ thread منفصل، وإلا لو ffmpeg
        # طبع كمية log أكبر من سعة الـ pipe (~64KB)، بيصير deadlock: ffmpeg
        # بيتعلق وهو عم يحاول يكتب على stderr الممتلئ، وبالتالي ما بيرجع
        # يقرأ فريمات جديدة من stdin، وبالتالي writer.write() بتضل معلّقة
        # لحد ما تنقتل العملية من برا — وهاد بالضبط شكل "الفيديو طلع فاضي".
        self._stderr_tail = []
        self._stderr_lock = threading.Lock()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()

    def _drain_stderr(self):
        try:
            for line in iter(self._proc.stderr.readline, b""):
                with self._stderr_lock:
                    self._stderr_tail.append(line.decode(errors="ignore").rstrip())
                    if len(self._stderr_tail) > self._STDERR_TAIL_LINES:
                        self._stderr_tail.pop(0)
        except (ValueError, OSError):
            pass  # الـ pipe انسكر، طبيعي عند الإغلاق

    def _get_stderr_tail(self):
        with self._stderr_lock:
            return "\n".join(self._stderr_tail)

    def write(self, frame):
        if self._closed:
            return

        # لازم تكون البيانات متسلسلة بالذاكرة (C-contiguous) وإلا tobytes()
        # ممكن يطلع ترتيب بايتات غلط لو الفريم ناتج عن slicing/crop
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        frame_bytes = frame.tobytes()
        if len(frame_bytes) != self._expected_frame_bytes:
            self._closed = True
            self._proc.stdin.close()
            self._proc.kill()
            raise ValueError(
                f"[FFmpegStreamWriter] حجم الفريم ({len(frame_bytes)} بايت) "
                f"ما بيطابق الأبعاد المتوقعة ({self._expected_frame_bytes} بايت). "
                f"تأكد إنو كل الفريمات بنفس أبعاد (w, h) يلي انبعتت لـ FFmpegStreamWriter."
            )

        try:
            self._proc.stdin.write(frame_bytes)
        except (BrokenPipeError, OSError) as exc:
            # لو ffmpeg وقع، اطلع الخطأ الحقيقي منه بدل ما يضيع بصمت
            self._closed = True
            raise RuntimeError(
                f"[FFmpegStreamWriter] ffmpeg process failed while writing frame: {exc}\n"
                f"ffmpeg stderr (آخر أسطر):\n{self._get_stderr_tail()}"
            )

    def release(self):
        if self._closed:
            return
        self._closed = True
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        return_code = self._proc.wait()
        self._stderr_thread.join(timeout=2)
        if return_code != 0:
            raise RuntimeError(
                f"[FFmpegStreamWriter] ffmpeg exited with code {return_code} "
                f"for {self._output_path}\nffmpeg stderr (آخر أسطر):\n{self._get_stderr_tail()}"
            )