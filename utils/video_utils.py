import cv2 as cv

def read_video(video_path):
    cap = cv.VideoCapture(video_path)

    fps = cap.get(cv.CAP_PROP_FPS)
    if not fps or fps != fps: 
        fps = 30.0

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    return frames, fps 

def save_video(output_frames, output_path, fps):
    if not output_frames:
        return
    height, width, _ = output_frames[0].shape
    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in output_frames:
        out.write(frame)
    out.release()


def get_video_fps(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f" Failed to open video: {video_path} ")
    fps = cap.get(cv.CAP_PROP_FPS)
    cap.release()
    if not fps or fps != fps:  
        fps = 30.0
    return fps


def iter_video_frames(video_path):
    
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f" Failed to open video: {video_path} ")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


class VideoWriterContext:
    

    def __init__(self, output_path, fps, frame_size):
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        self._writer = cv.VideoWriter(output_path, fourcc, fps, frame_size)
        if not self._writer.isOpened():
            raise IOError(f"Failed to create output video file: {output_path}")

    def write(self, frame):
        self._writer.write(frame)

    def release(self):
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()