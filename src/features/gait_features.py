import numpy as np

# MediaPipe landmark indices (RIGHT leg)
HIP = 24
KNEE = 26
ANKLE = 28
HEEL = 30
FOOT = 32


# Utilities
def smooth_signal(signal, window=7):
    if len(signal) < window:
        return signal
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


def angle_3points(a, b, c):
    """Angle at point b (degrees)"""
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

# Gait Feature Extraction
def extract_gait_features(keypoints: np.ndarray, fps: float = 30.0):
   
    # keypoints: (T, 33, 4) from MediaPipe
    # Returns dictionary of gait features
    ankle_angles = []
    knee_angles = []

    foot_y = []
    ankle_y = []

    for frame in keypoints:
        hip = frame[HIP][:3]
        knee = frame[KNEE][:3]
        ankle = frame[ANKLE][:3]
        foot = frame[FOOT][:3]

        if (
            np.any(np.isnan(hip)) or
            np.any(np.isnan(knee)) or
            np.any(np.isnan(ankle)) or
            np.any(np.isnan(foot))
        ):
            continue

        if (
            np.linalg.norm(knee - ankle) < 1e-4 or
            np.linalg.norm(foot - ankle) < 1e-4 or
            np.linalg.norm(hip - knee) < 1e-4
        ):
            continue

        ankle_angles.append(angle_3points(knee, ankle, foot))
        knee_angles.append(angle_3points(hip, knee, ankle))

        # invert y (image coords → ground-up)
        foot_y.append(1.0 - foot[1])
        ankle_y.append(1.0 - ankle[1])

    if len(ankle_y) < fps:
        return _empty_features()

    ankle_angles = smooth_signal(np.array(ankle_angles))
    knee_angles  = smooth_signal(np.array(knee_angles))
    foot_y       = smooth_signal(np.array(foot_y))
    ankle_y      = smooth_signal(np.array(ankle_y))

    # STEP & GAIT EVENT DETECTION
    # Vertical ankle velocity
    vel = smooth_signal(np.diff(ankle_y), window=7)

    # Heel strike: ankle reaches local minimum
    heel_strikes = np.where(
        (vel[:-1] < 0) & (vel[1:] > 0)
    )[0] + 1

    # Toe-off: ankle reaches local maximum
    toe_offs = np.where(
        (vel[:-1] > 0) & (vel[1:] < 0)
    )[0] + 1

    # Build gait cycles
    stance_ratios = []
    step_heights = []

    for i in range(len(heel_strikes) - 1):
        hs = heel_strikes[i]
        next_hs = heel_strikes[i + 1]

        # toe-off must occur between heel strikes
        tos = toe_offs[(toe_offs > hs) & (toe_offs < next_hs)]
        if len(tos) == 0:
            continue

        to = tos[0]

        cycle_time = (next_hs - hs) / fps
        stance_time = (to - hs) / fps

        if cycle_time <= 0:
            continue

        stance_ratios.append(stance_time / cycle_time)

        step_heights.append(
            np.max(foot_y[hs:next_hs]) - np.min(foot_y[hs:next_hs])
        )

    # Cadence
    duration_sec = len(ankle_y) / fps
    cadence = (len(heel_strikes) / duration_sec) * 60 if duration_sec > 0 else 0

    # Aggregate features
    stance_ratio = float(np.mean(stance_ratios)) if stance_ratios else 0.0
    step_height  = float(np.mean(step_heights)) if step_heights else 0.0

    features = {
        "mean_ankle_angle": float(np.mean(ankle_angles)),
        "ankle_angle_range": float(np.max(ankle_angles) - np.min(ankle_angles)),
        "ankle_angle_std": float(np.std(ankle_angles)),

        "mean_knee_angle": float(np.mean(knee_angles)),
        "knee_angle_range": float(np.max(knee_angles) - np.min(knee_angles)),
        "knee_angle_std": float(np.std(knee_angles)),

        "mean_step_height": step_height,
        "cadence": float(cadence),
        "stance_ratio": stance_ratio,
    }

    return _sanitize(features)

# Helpers
def _sanitize(features: dict):
    clean = {}
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v):
            clean[k] = 0.0
        else:
            clean[k] = float(v)
    return clean

def _empty_features():
    return {
        "mean_ankle_angle": 0.0,
        "ankle_angle_range": 0.0,
        "ankle_angle_std": 0.0,
        "mean_knee_angle": 0.0,
        "knee_angle_range": 0.0,
        "knee_angle_std": 0.0,
        "mean_step_height": 0.0,
        "cadence": 0.0,
        "stance_ratio": 0.0,
    }
