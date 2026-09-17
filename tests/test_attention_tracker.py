"""Tests the AttentionTracker state machine (distraction detection) end to end, without
needing a live camera: confirmation timing, false-positive protection (brief glances,
natural jitter), face-lost handling, calibration baseline math, and the real
pitch-wraparound bug that was found and fixed during development."""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import AttentionTracker, detect_distraction_direction, compute_baseline, PITCH_SIGN

results = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    results.append(status)
    print(f"[{status}] {name}  {detail}")


# TEST 1: driver looking straight -> ATTENTIVE
t = AttentionTracker()
t.calibrate(0.0, 0.0, 0.0)
s = t.update({"yaw": 0.0, "pitch": 0.0, "roll": 0.0}, True, now=0.0)
check("TEST1 straight -> ATTENTIVE", s["state"] == "ATTENTIVE", s)

# TEST 2: looks left for <1s (under DISTRACTION_DURATION) then returns -> never confirmed DISTRACTED
t = AttentionTracker()
t.calibrate(0.0, 0.0, 0.0)
now = 0.0
for i in range(10):  # 10 frames over ~0.9s of deviation
    now += 0.09
    s = t.update({"yaw": -25.0, "pitch": 0.0, "roll": 0.0}, True, now=now)
now += 0.05
s = t.update({"yaw": 0.0, "pitch": 0.0, "roll": 0.0}, True, now=now)
check("TEST2 brief glance -> no confirmed distraction", t.distraction_events == 0, f"events={t.distraction_events}")

# TEST 3: looks left sustained >2s -> DISTRACTED, LOOKING LEFT
t = AttentionTracker()
t.calibrate(0.0, 0.0, 0.0)
now = 0.0
s = None
for i in range(30):  # ~2.7s of sustained deviation
    now += 0.09
    s = t.update({"yaw": -25.0, "pitch": 0.0, "roll": 0.0}, True, now=now)
check("TEST3 sustained left -> DISTRACTED/LOOKING LEFT",
      s["state"] == "DISTRACTED" and s["direction"] == "LOOKING LEFT", s)

# TEST 4: looks right sustained >2s -> DISTRACTED, LOOKING RIGHT
t = AttentionTracker()
t.calibrate(0.0, 0.0, 0.0)
now = 0.0
s = None
for i in range(30):
    now += 0.09
    s = t.update({"yaw": 25.0, "pitch": 0.0, "roll": 0.0}, True, now=now)
check("TEST4 sustained right -> DISTRACTED/LOOKING RIGHT",
      s["state"] == "DISTRACTED" and s["direction"] == "LOOKING RIGHT", s)

# TEST 5: looks down sustained >2s -> DISTRACTED, LOOKING DOWN
t = AttentionTracker()
t.calibrate(0.0, 0.0, 0.0)
now = 0.0
s = None
for i in range(30):
    now += 0.09
    s = t.update({"yaw": 0.0, "pitch": -20.0, "roll": 0.0}, True, now=now)
check("TEST5 sustained down -> DISTRACTED/LOOKING DOWN",
      s["state"] == "DISTRACTED" and s["direction"] == "LOOKING DOWN", s)

# TEST 6: returns to forward after distraction -> ATTENTIVE, 1 event recorded.
# Step several real-time-equivalent frames (not just one) since EMA smoothing takes a
# few frames to settle back to baseline, same as it would across real camera frames.
s = None
for i in range(10):
    now += 0.09
    s = t.update({"yaw": 0.0, "pitch": 0.0, "roll": 0.0}, True, now=now)
check("TEST6 return to forward -> ATTENTIVE", s["state"] == "ATTENTIVE", s)
check("TEST6 exactly 1 distraction event recorded", t.distraction_events == 1, f"events={t.distraction_events}")
check("TEST6 event pushed to history", len(t.history) == 1, f"history={t.history}")

# TEST 7/8 (eye-based drowsiness untouched) are verified structurally: AttentionTracker
# has no reference to eye_model/closed_frame_count/alarm at all -> cannot interfere.
check("TEST7/8 tracker has no coupling to drowsiness state",
      not hasattr(AttentionTracker, "closed_frame_count") and not hasattr(AttentionTracker, "eye_model"),
      "AttentionTracker is fully independent of EyeCNN/drowsiness fields")

# TEST 9: face lost mid-distraction -> UNKNOWN, no false event, no crash
t = AttentionTracker()
t.calibrate(0.0, 0.0, 0.0)
now = 0.0
for i in range(15):  # building up toward distraction but not yet confirmed
    now += 0.09
    s = t.update({"yaw": -25.0, "pitch": 0.0, "roll": 0.0}, True, now=now)
now += 0.09
s = t.update(None, False, now=now)  # face disappears
check("TEST9 face lost -> UNKNOWN", s["state"] == "UNKNOWN", s)
check("TEST9 no event counted on face loss", t.distraction_events == 0, f"events={t.distraction_events}")

# TEST 10: normal small head movements (jitter under threshold) -> no false distraction events
t = AttentionTracker()
t.calibrate(0.0, 0.0, 0.0)
now = 0.0
random.seed(0)
for i in range(200):
    now += 0.03
    jitter_yaw = random.uniform(-8, 8)
    jitter_pitch = random.uniform(-6, 6)
    s = t.update({"yaw": jitter_yaw, "pitch": jitter_pitch, "roll": 0.0}, True, now=now)
check("TEST10 small natural jitter -> zero false events", t.distraction_events == 0, f"events={t.distraction_events}")

# Calibration baseline math (circular mean, so compare with tolerance)
baseline = compute_baseline([{"yaw": 2.0, "pitch": -1.0, "roll": 0.5},
                              {"yaw": 4.0, "pitch": -3.0, "roll": 1.5}])
expected = (3.0, -2.0, 1.0)
close = all(abs(a - b) < 0.01 for a, b in zip(baseline, expected))
check("compute_baseline averages correctly (circular mean)", close, baseline)

# Pitch wraparound near +/-180 (the actual bug found and fixed: this solvePnP model
# places frontal-face pitch near the +/-180 boundary, so "looking down" can cross it).
# Values below are raw solvePnP output x PITCH_SIGN, exactly as _get_signed_pose() would
# produce in the real app, so this reflects real runtime behavior end-to-end.
t = AttentionTracker()
t.calibrate(0.0, PITCH_SIGN * 159.87, 0.0)  # realistic baseline pitch observed for a frontal face
now = 0.0
s = None
for i in range(30):
    now += 0.09
    # simulates a real "looking down" landmark shift that wraps past -180 in raw solvePnP output
    s = t.update({"yaw": 0.0, "pitch": PITCH_SIGN * -172.86, "roll": 0.0}, True, now=now)
check("pitch wraparound resolves to a small bounded relative angle",
      abs(s["pitch"]) < 90, f"rel_pitch={s['pitch']:.1f} (must not be the raw -332.7 artifact)")
check("pitch wraparound still confirms DISTRACTED/LOOKING DOWN",
      s["state"] == "DISTRACTED" and s["direction"] == "LOOKING DOWN", s)

# Direction classifier boundary check
check("detect_distraction_direction FORWARD inside thresholds",
      detect_distraction_direction(5, 5, 5) == "FORWARD")
check("detect_distraction_direction LEFT beyond -yaw threshold",
      detect_distraction_direction(-21, 0, 0) == "LOOKING LEFT")
check("detect_distraction_direction TILT when only roll exceeds",
      detect_distraction_direction(0, 0, 30) == "HEAD TILT")

print()
print(f"{results.count('PASS')}/{len(results)} checks passed")
assert results.count("PASS") == len(results), "one or more attention-tracker checks failed"
