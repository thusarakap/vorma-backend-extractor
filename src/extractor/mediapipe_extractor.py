import cv2
import mediapipe as mp
import json
import numpy as np
import os
import argparse

mp_pose = mp.solutions.pose

def extract_pose_from_video(video_path: str, out_dir: str):
    print(f"[INFO] Input video: {video_path}")
    print(f"[INFO] Output dir: {out_dir}")

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found at: {video_path}")
        return

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Could not open video file.")
        return

    frames = []
    idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        print("[INFO] Starting frame processing...")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(frame_rgb)

            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                points = [[lm.x, lm.y, lm.z, lm.visibility] for lm in lms]
            else:
                points = None

            frames.append({"frame": idx, "landmarks": points})
            idx += 1

            if idx % 50 == 0:
                print(f"[INFO] Processed {idx} frames...")

    cap.release()

    print(f"[INFO] Total frames processed: {idx}")

    # Save to JSON
    json_file = os.path.join(out_dir, "keypoints.json")
    with open(json_file, "w") as f:
        json.dump(frames, f, indent=2)

    # Save to NumPy
    n_landmarks = 33
    arr = np.full((len(frames), n_landmarks, 4), np.nan, dtype=np.float32)
    for i, frame in enumerate(frames):
        if frame["landmarks"]:
            arr[i] = np.array(frame["landmarks"])

    np_file = os.path.join(out_dir, "keypoints.npy")
    np.save(np_file, arr)

    print("[OK] Saved:")
    print("   JSON :", json_file)
    print("   Numpy:", np_file)

    return json_file, np_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to input video")
    parser.add_argument(
        "--out",
        default="output",
        help="Directory to save extracted keypoints",
    )
    args = parser.parse_args()

    extract_pose_from_video(args.video, args.out)

if __name__ == "__main__":
    main()
