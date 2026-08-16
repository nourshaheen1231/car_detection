import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from bridge.runner import run_ai_job

app = FastAPI(title="AI Bridge")

class ProcessRequest(BaseModel):
    video_id: int
    video_path: str

@app.post("/process-video")
async def process_video(req: ProcessRequest, background_tasks: BackgroundTasks):
    if not os.path.exists(req.video_path):
        return {"status": "error", "message": "Video file not found"}

    background_tasks.add_task(run_ai_job, req.video_id, req.video_path)
    return {"status": "started", "video_id": req.video_id}

@app.get("/health")
async def health():
    return {"status": "ok"}