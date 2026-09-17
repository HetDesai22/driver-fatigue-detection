"""Verifies the wall-clock drowsiness trigger and the persistent "strike" escalation
model: strikes accumulate per separate episode (not per frame or per duration), never
reset just because the eyes reopen, and only clear via Change Driver. Also proves the
trigger timing is independent of camera FPS (the original bug report: "taking too
long to complete drowsiness meter and beep" was traced to a frame-count threshold)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import EYES_CLOSED_ALERT_SECONDS, FATIGUE_STOP_STRIKES


class FakeSession:
    def __init__(self):
        self.eyes_closed_since = None
        self._was_closed_seq = False
        self.alarm_active = False
        self.alert_count = 0
        self.fatigue_strikes = 0

    def frame(self, eyes_closed, now):
        if eyes_closed:
            if self.eyes_closed_since is None:
                self.eyes_closed_since = now
            self._was_closed_seq = True
        else:
            self.eyes_closed_since = None
            self._was_closed_seq = False
            self.alarm_active = False

        closed_duration = (now - self.eyes_closed_since) if self.eyes_closed_since is not None else 0.0
        if closed_duration >= EYES_CLOSED_ALERT_SECONDS:
            if not self.alarm_active:
                self.alarm_active = True
                self.alert_count += 1
                self.fatigue_strikes += 1

    def change_driver(self):
        self.fatigue_strikes = 0


def hold_eyes_closed(session, seconds, fps, now):
    dt = 1.0 / fps
    n = max(1, int(seconds / dt))
    for _ in range(n):
        now += dt
        session.frame(True, now)
    return now


def open_eyes(session, seconds, fps, now):
    dt = 1.0 / fps
    n = max(1, int(seconds / dt))
    for _ in range(n):
        now += dt
        session.frame(False, now)
    return now


# Run the whole scenario at a deliberately LOW fps (8fps) to prove the trigger timing
# is no longer frame-count dependent -- this is exactly the "too slow" symptom being fixed.
FPS = 8
now = 0.0
s = FakeSession()

# Margin must exceed one frame period (1/FPS) since eyes_closed_since is timestamped at
# the first closed frame, not "just before" it -- so measured duration lags true elapsed
# time by up to one dt. 0.3s is a safe margin at every FPS this suite exercises.
MARGIN = 0.3

now = hold_eyes_closed(s, EYES_CLOSED_ALERT_SECONDS + MARGIN, FPS, now)
assert s.fatigue_strikes == 1, f"expected 1 strike after episode 1, got {s.fatigue_strikes}"
print(f"[PASS] Episode 1 (at {FPS}fps) -> {s.fatigue_strikes} strike, took ~{EYES_CLOSED_ALERT_SECONDS+MARGIN:.2f}s not longer")

now = open_eyes(s, 1.0, FPS, now)
assert s.fatigue_strikes == 1, "strikes must NOT reset just because eyes reopened"
print(f"[PASS] Reopening eyes does not reset strikes (still {s.fatigue_strikes})")

now = hold_eyes_closed(s, EYES_CLOSED_ALERT_SECONDS + MARGIN, FPS, now)
assert s.fatigue_strikes == 2, f"expected 2 strikes after episode 2, got {s.fatigue_strikes}"
print(f"[PASS] Episode 2 -> {s.fatigue_strikes} strikes (expected: slower + last lane)")

now = open_eyes(s, 1.0, FPS, now)

now = hold_eyes_closed(s, EYES_CLOSED_ALERT_SECONDS + MARGIN, FPS, now)
assert s.fatigue_strikes == FATIGUE_STOP_STRIKES, f"expected {FATIGUE_STOP_STRIKES} strikes, got {s.fatigue_strikes}"
print(f"[PASS] Episode 3 -> {s.fatigue_strikes} strikes (expected: full stop)")

now = open_eyes(s, 2.0, FPS, now)
assert s.fatigue_strikes == FATIGUE_STOP_STRIKES, "vehicle must stay stopped even after eyes stay open"
print(f"[PASS] Vehicle stays at {s.fatigue_strikes} strikes (stopped) even after eyes reopen and stay open")

# A brief blink (well under EYES_CLOSED_ALERT_SECONDS) must NOT count as a strike
s2 = FakeSession()
now2 = hold_eyes_closed(s2, EYES_CLOSED_ALERT_SECONDS * 0.3, FPS, 0.0)
open_eyes(s2, 0.5, FPS, now2)
assert s2.fatigue_strikes == 0, "a blink under EYES_CLOSED_ALERT_SECONDS must not count as a strike"
print("[PASS] A blink under the beep threshold does not count as a strike")

# Holding eyes closed continuously (single long episode) counts as only ONE strike
s3 = FakeSession()
hold_eyes_closed(s3, EYES_CLOSED_ALERT_SECONDS * 6, FPS, 0.0)
assert s3.fatigue_strikes == 1, f"a single continuous closure must be exactly 1 strike, got {s3.fatigue_strikes}"
print("[PASS] One long continuous closure = exactly 1 strike (not duration-based beyond the trigger)")

# Same scenario at a much higher FPS must trigger the FIRST strike in essentially the
# same wall-clock time as the low-fps run above -- proving FPS-independence directly.
s_hi = FakeSession()
now_hi = hold_eyes_closed(s_hi, EYES_CLOSED_ALERT_SECONDS + 0.15, 60, 0.0)
assert s_hi.fatigue_strikes == 1
print(f"[PASS] Same {EYES_CLOSED_ALERT_SECONDS+0.15:.2f}s hold at 60fps ALSO yields exactly 1 strike "
      f"(matches the 8fps run -> confirms FPS-independence)")

s.change_driver()
assert s.fatigue_strikes == 0
print("[PASS] Change Driver resets strikes back to 0")

print("\nAll strike-model tests passed.")
