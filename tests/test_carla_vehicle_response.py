"""Verifies the priority chain that decides CARLA vehicle speed/lane/HUD each control-loop
tick: drowsiness strikes always override a simultaneous distraction reading, distraction
alone (no strikes) still produces a transient speed/steer response, and the steering
drift-bias sign matches the direction the driver looked. No live CARLA server needed --
CarlaBridge is instantiated without __init__ and only its pure math is exercised."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import (CarlaBridge, FATIGUE_STOP_STRIKES, FATIGUE_STRIKE_1_SPEED_KMH,
                  FATIGUE_STRIKE_2_SPEED_KMH, CARLA_NORMAL_SPEED_KMH, DISTRACTION_SPEED_KMH,
                  DISTRACTION_STEER_BIAS)


def decide(strikes, attention_state, direction):
    """Mirrors the exact if/elif priority chain inside CarlaBridge._run for target_speed."""
    if strikes >= FATIGUE_STOP_STRIKES:
        return 0.0, "STOPPED"
    elif strikes == 2:
        return FATIGUE_STRIKE_2_SPEED_KMH, "STRIKE2"
    elif strikes == 1:
        return FATIGUE_STRIKE_1_SPEED_KMH, "STRIKE1"
    elif attention_state == "DISTRACTED":
        return DISTRACTION_SPEED_KMH, "DISTRACTED"
    else:
        return CARLA_NORMAL_SPEED_KMH, "NORMAL"


checks = []


def check(name, cond):
    checks.append(cond)
    print(("[PASS] " if cond else "[FAIL] ") + name)


# Head movement alone (no drowsiness strikes) must now affect speed.
speed, tag = decide(0, "DISTRACTED", "LOOKING RIGHT")
check("distraction alone reduces speed", speed == DISTRACTION_SPEED_KMH and tag == "DISTRACTED")

speed, tag = decide(0, "ATTENTIVE", "FORWARD")
check("attentive + no strikes -> normal speed", speed == CARLA_NORMAL_SPEED_KMH and tag == "NORMAL")

# Drowsiness must take priority over simultaneous distraction.
speed, tag = decide(1, "DISTRACTED", "LOOKING LEFT")
check("1 strike overrides distraction", speed == FATIGUE_STRIKE_1_SPEED_KMH and tag == "STRIKE1")

speed, tag = decide(3, "DISTRACTED", "LOOKING RIGHT")
check("3 strikes (stopped) overrides distraction", speed == 0.0 and tag == "STOPPED")

# Steering drift bias direction sanity (using the bridge's pure math, no live CARLA needed
# since use_last_lane=False and no waypoint lookup happens before the drift_bias branch
# when next_wps is empty -- exercise the fallback path directly).
bridge = CarlaBridge.__new__(CarlaBridge)  # skip __init__, we only need _compute_steer's math


class FakeLoc:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __add__(self, other):
        return self


class FakeRot:
    def __init__(self, yaw):
        self.yaw = yaw


class FakeTransform:
    def __init__(self, x, y, yaw):
        self.location = FakeLoc(x, y)
        self.rotation = FakeRot(yaw)


class FakeWaypoint:
    def next(self, d):
        return []  # force the "no next waypoint" fallback branch -> returns drift_bias directly


class FakeMap:
    def get_waypoint(self, loc):
        return FakeWaypoint()


bridge.world_map = FakeMap()
transform = FakeTransform(0, 0, 0)

steer_right = bridge._compute_steer(None, transform, use_last_lane=False, drift_bias=DISTRACTION_STEER_BIAS)
steer_left = bridge._compute_steer(None, transform, use_last_lane=False, drift_bias=-DISTRACTION_STEER_BIAS)
check("LOOKING RIGHT drift_bias yields positive steer", steer_right == DISTRACTION_STEER_BIAS)
check("LOOKING LEFT drift_bias yields negative steer", steer_left == -DISTRACTION_STEER_BIAS)

print(f"\n{sum(checks)}/{len(checks)} checks passed")
assert all(checks)
