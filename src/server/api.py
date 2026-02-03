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

# from src.models.synthetic_pressure import generate_pressure_map
# from src.models.cnn_model import predict_load_from_pressure
# from src.models.prescription import load_to_prescription


app = FastAPI(title="VORMA Prototype API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


# @app.post("/api/analyze")
# async def analyze_video(file: UploadFile = File(...)):
#     """
#     1. Save uploaded video
#     2. Run Mediapipe extractor -> keypoints.npy
#     3. Run simple model -> height map
#     4. Return JSON with results + basic metadata
#     """
#     ext = Path(file.filename).suffix or ".mp4"
#     vid_id = uuid.uuid4().hex

#     save_path = UPLOAD_DIR / f"{vid_id}{ext}"
#     with save_path.open("wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     # Run extractor
#     out_dir = PROCESSED_DIR / vid_id
#     json_path, npy_path = extract_pose_from_video(str(save_path), str(out_dir))

#     # Load keypoints
#     keypoints = np.load(npy_path)  # (T, 33, 4)

#     # 1. Extract gait features
#     gait_features = extract_gait_features(keypoints)

#     # 2. Generate synthetic plantar pressure map
#     pressure_map = generate_pressure_map(gait_features)

#     # 3. CNN prediction (plantar load)
#     predicted_load = predict_load_from_pressure(pressure_map)

#     # 4. Orthotic prescription
#     prescription = load_to_prescription(predicted_load)

#     response = {
#         "id": vid_id,
#         "video_filename": file.filename,
#         "num_frames": int(keypoints.shape[0]),
#         "gait_features": gait_features,
#         "predicted_load": predicted_load,
#         "orthotic_prescription": prescription
#     }

#    return JSONResponse(response)
