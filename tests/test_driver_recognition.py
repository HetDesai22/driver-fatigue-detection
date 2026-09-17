"""Verifies the LBPH driver-recognition pipeline used for automatic "driver changed"
detection: same-driver frames match below threshold, a different driver's face matches
above threshold, and the consecutive-mismatch debounce ignores a single noisy frame
while still triggering on a sustained mismatch. Uses synthetic textured images (LBPH
cares about local texture patterns, not real facial structure) so this runs without a
camera; real-world accuracy still depends on your actual lighting/camera (see README)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
from app import DRIVER_CHANGE_LBPH_THRESHOLD, DRIVER_CHANGE_CONFIRM_COUNT, MIN_CALIB_FACE_SAMPLES

checks = []


def check(name, cond, detail=""):
    checks.append(cond)
    print(("[PASS] " if cond else "[FAIL] ") + name, detail)


def make_face(seed, size=200):
    rng = np.random.RandomState(seed)
    base = rng.randint(80, 180, (size, size), dtype=np.uint8)
    for _ in range(8):
        x, y = rng.randint(0, size - 40, 2)
        base[y:y + 40, x:x + 40] = rng.randint(0, 255)
    return base


def jitter(img, seed):
    """Simulate a slightly different frame of the SAME person (minor noise/lighting)."""
    rng = np.random.RandomState(seed)
    noise = rng.randint(-10, 10, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# Train on driver A's reference samples (mirrors calibration capture)
driver_a_ref = [jitter(make_face(seed=1), seed=100 + i) for i in range(MIN_CALIB_FACE_SAMPLES + 2)]
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(driver_a_ref, np.array([0] * len(driver_a_ref)))

# Same driver, new frame -> should be a good match (low confidence/distance)
same_driver_frame = jitter(make_face(seed=1), seed=999)
_, conf_same = recognizer.predict(same_driver_frame)
check("same driver yields confidence below threshold",
      conf_same <= DRIVER_CHANGE_LBPH_THRESHOLD, f"confidence={conf_same:.1f}")

# A different driver's face -> should be a poor match (high confidence/distance)
driver_b_frame = make_face(seed=42)
_, conf_diff = recognizer.predict(driver_b_frame)
check("different driver yields confidence ABOVE threshold",
      conf_diff > DRIVER_CHANGE_LBPH_THRESHOLD, f"confidence={conf_diff:.1f}")

check("different-driver confidence is clearly higher than same-driver confidence",
      conf_diff > conf_same, f"same={conf_same:.1f} diff={conf_diff:.1f}")


def simulate_mismatch_sequence(confidences):
    """Mirrors the exact debounce check in _update_frame."""
    mismatch_count = 0
    triggered_at = None
    for i, c in enumerate(confidences):
        if c > DRIVER_CHANGE_LBPH_THRESHOLD:
            mismatch_count += 1
        else:
            mismatch_count = 0
        if mismatch_count >= DRIVER_CHANGE_CONFIRM_COUNT and triggered_at is None:
            triggered_at = i
    return triggered_at


seq = [30, 30, 95, 30, 30, 30]
trig = simulate_mismatch_sequence(seq)
check("a single noisy mismatch frame does not falsely trigger driver-change", trig is None, f"triggered_at={trig}")

seq2 = [30, 30, 95, 95, 95, 30]
trig2 = simulate_mismatch_sequence(seq2)
check(f"sustained mismatch triggers after {DRIVER_CHANGE_CONFIRM_COUNT} consecutive frames",
      trig2 == 4, f"triggered_at={trig2}")

print(f"\n{sum(checks)}/{len(checks)} checks passed")
assert all(checks)
