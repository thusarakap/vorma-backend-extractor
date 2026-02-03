import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

print("[DEBUG] overlay_skeleton.py started")

# Hard-coded paths for now – easier to debug
video_path = Path("data/videos/walk1.mp4")
keypoints_path = Path("data/processed/walk1/keypoints.npy")
out_path = Path("overlay_walk1.avi")

print(f"[DEBUG] Video path:    {video_path}")
print(f"[DEBUG] Keypoints path:{keypoints_path}")

if not video_path.exists():
    print("[ERROR] Video file does not exist.")
    raise SystemExit(1)

if not keypoints_path.exists():
    print("[ERROR] Keypoints file does not exist.")
    raise SystemExit(1)

# Load keypoints
arr = np.load(str(keypoints_path))  # (T, 33, 4)
num_frames_arr = arr.shape[0]
print(f"[DEBUG] Loaded keypoints, shape: {arr.shape}")

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print("[ERROR] Could not open video file.")
    raise SystemExit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 1 or np.isnan(fps):
    print("[WARN] Invalid FPS from video, defaulting to 30")
    fps = 30.0

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[DEBUG] Video resolution: {w}x{h} @ {fps} fps")

# Force AVI+MJPG – very compatible
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out_path = out_path.with_suffix(".avi")

writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
if not writer.isOpened():
    print("[ERROR] Could not open VideoWriter.")
    cap.release()
    raise SystemExit(1)

mp_pose = mp.solutions.pose
connections = list(mp_pose.POSE_CONNECTIONS)

print("[INFO] Generating overlay video...")
frame_idx = 0
frames_written = 0

while True:
    ok, frame = cap.read()
    if not ok:
        print("[DEBUG] No more frames from video.")
        break

    if frame_idx >= num_frames_arr:
        print("[DEBUG] No more keypoints frames, stopping.")
        break

    pts = arr[frame_idx]  # (33, 4)

    if not np.isnan(pts).all():
        lm_list = []
        for (x, y, z, _v) in pts:
            px = int(x * w)
            py = int(y * h)
            lm_list.append((px, py))

        for a, b in connections:
            try:
                xa, ya = lm_list[a]
                xb, yb = lm_list[b]
                cv2.line(frame, (xa, ya), (xb, yb), (0, 255, 0), 2)
            except IndexError:
                continue

    writer.write(frame)
    frames_written += 1
    frame_idx += 1

    if frames_written % 100 == 0:
        print(f"[INFO] Written {frames_written} frames...")

cap.release()
writer.release()

print(f"[OK] Finished. Frames written: {frames_written}")
print(f"[OK] Saved overlay video: {out_path}")
