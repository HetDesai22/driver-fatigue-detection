"""Sanity-checks the solvePnP head-pose math in isolation, using synthetic landmark
positions instead of a live camera. Verifies a symmetric frontal face yields near-zero
yaw/roll, and that shifting the nose asymmetrically changes the computed yaw."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import estimate_head_pose


class P:
    def __init__(self, x, y):
        self.x, self.y = x, y


def make_landmarks(nose, chin, l_eye, r_eye, l_mouth, r_mouth):
    lm = [P(0.5, 0.5)] * 300
    lm = list(lm)
    lm[1] = P(*nose)
    lm[152] = P(*chin)
    lm[33] = P(*l_eye)
    lm[263] = P(*r_eye)
    lm[61] = P(*l_mouth)
    lm[291] = P(*r_mouth)
    return lm


W, H = 640, 480

lm_frontal = make_landmarks(
    nose=(0.50, 0.55), chin=(0.50, 0.85),
    l_eye=(0.35, 0.45), r_eye=(0.65, 0.45),
    l_mouth=(0.40, 0.68), r_mouth=(0.60, 0.68),
)
pose = estimate_head_pose(lm_frontal, W, H)
print("Frontal symmetric face pose:", pose)
assert pose is not None, "solvePnP failed to converge on a valid symmetric input"
assert abs(pose["yaw"]) < 10, f"expected near-zero yaw for symmetric face, got {pose['yaw']}"
assert abs(pose["roll"]) < 10, f"expected near-zero roll for symmetric face, got {pose['roll']}"
print("[PASS] frontal face yields near-zero yaw/roll")

lm_turned = make_landmarks(
    nose=(0.58, 0.55), chin=(0.50, 0.85),
    l_eye=(0.35, 0.45), r_eye=(0.65, 0.45),
    l_mouth=(0.40, 0.68), r_mouth=(0.60, 0.68),
)
pose_turned = estimate_head_pose(lm_turned, W, H)
print("Nose-shifted-right pose:", pose_turned)
assert pose_turned is not None
assert pose_turned["yaw"] != pose["yaw"], "yaw should respond to nose asymmetry"
print("[PASS] yaw changes when nose position shifts asymmetrically")

print("\nAll head-pose math sanity checks passed.")
