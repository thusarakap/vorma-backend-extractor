import numpy as np

# MediaPipe landmark indices (RIGHT leg)
HIP = 24
KNEE = 26
ANKLE = 28
HEEL = 30
FOOT = 32

def smooth_signal(signal, window=5):
    if len(signal) < window:
        return signal
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")

def angle_3points(a, b, c):
    """
    Compute angle at point b (in degrees) for points a-b-c
    """
    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle


def extract_gait_features(keypoints: np.ndarray, fps: float = 30.0):
    """
    keypoints: (T, 33, 4) from MediaPipe
    Returns dictionary of gait features
    """

    ankle_angles = []
    knee_angles = []
    foot_heights = []

    ankle_y = []
    
    valid_frames = 0

    for frame in keypoints:
        # Extract xyz only
        hip = frame[HIP][:3]
        knee = frame[KNEE][:3]
        ankle = frame[ANKLE][:3]
        foot = frame[FOOT][:3]

        # Skip frames with NaNs
        if (
            np.any(np.isnan(hip)) or
            np.any(np.isnan(knee)) or
            np.any(np.isnan(ankle)) or
            np.any(np.isnan(foot))
        ):
            continue

        # Skip degenerate geometry
        if (
            np.linalg.norm(knee - ankle) < 1e-4 or
            np.linalg.norm(foot - ankle) < 1e-4 or
            np.linalg.norm(hip - knee) < 1e-4
        ):
            continue

        ankle_angles.append(angle_3points(knee, ankle, foot))
        knee_angles.append(angle_3points(hip, knee, ankle))

        # Invert y-axis (image coordinates)
        foot_heights.append(1.0 - foot[1])
        ankle_y.append(1.0 - ankle[1])

        valid_frames += 1

    print(f"[DEBUG] Valid frames used: {valid_frames}")

    if valid_frames == 0:

        return {
            "mean_ankle_angle": 0.0,
            "std_ankle_angle": 0.0,
            "mean_knee_angle": 0.0,
            "mean_step_height": 0.0,
            "cadence": 0.0,
            "stance_ratio": 0.0,
        }


    ankle_angles = np.array(ankle_angles)
    knee_angles = np.array(knee_angles)
    foot_heights = np.array(foot_heights)
    ankle_y = np.array(ankle_y)

    ankle_angles = smooth_signal(np.array(ankle_angles))
    knee_angles  = smooth_signal(np.array(knee_angles))
    foot_heights = smooth_signal(np.array(foot_heights))
    ankle_y      = smooth_signal(np.array(ankle_y))

    # Step height (vertical range)
    step_height = np.max(foot_heights) - np.min(foot_heights)

    ankle_y = np.array(ankle_y)

    velocity = smooth_signal(np.diff(ankle_y), window=7)

    # Detect step peaks
    peaks = np.where(
        (velocity[:-1] > 0) & (velocity[1:] < 0)
    )[0]

    steps = len(peaks)
    duration_sec = len(ankle_y) / fps
    cadence = (steps / duration_sec) * 60 if duration_sec > 0 else 0

    # Stance ratio proxy
    # stance_frames = np.sum(np.abs(velocity) < np.percentile(np.abs(velocity), 25))
    # stance_ratio = stance_frames / len(velocity)
    contact_threshold = np.percentile(ankle_y, 20)
    stance_frames = np.sum(ankle_y < contact_threshold)
    stance_ratio = stance_frames / len(ankle_y)

    features = {
    "mean_ankle_angle": np.mean(ankle_angles),
    "ankle_angle_range": np.max(ankle_angles) - np.min(ankle_angles),
    "ankle_angle_std": np.std(ankle_angles),
    "mean_knee_angle": np.mean(knee_angles),
    "knee_angle_range": np.max(knee_angles) - np.min(knee_angles),
    "knee_angle_std": np.std(knee_angles),
    "mean_step_height": step_height,
    "cadence": cadence,
    "stance_ratio": stance_ratio,
    }

    # 🔧 SANITIZE VALUES FOR JSON
    clean_features = {}
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v):
            clean_features[k] = 0.0
        else:
            clean_features[k] = float(v)

    return clean_features