from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import uuid
import numpy as np
import os

from src.extractor.mediapipe_extractor import extract_pose_from_video
from src.features.gait_features import extract_gait_features

app = FastAPI(title="VORMA Prototype API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # allow all origins
    allow_credentials=True,
    allow_methods=["*"],            # allow all HTTP methods
    allow_headers=["*"],            # allow all headers
)

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
def root():
    return {"message": "VORMA Prototype API is running"}

@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix or ".mp4"
    vid_id = uuid.uuid4().hex

    save_path = UPLOAD_DIR / f"{vid_id}{ext}"
    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    out_dir = PROCESSED_DIR / vid_id
    _, npy_path = extract_pose_from_video(str(save_path), str(out_dir))

    keypoints = np.load(npy_path)

    gait_features = extract_gait_features(keypoints)

    return JSONResponse({
        "id": vid_id,
        "video_filename": file.filename,
        "num_frames": int(keypoints.shape[0]),
        "gait_features": gait_features
    })