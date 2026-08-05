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