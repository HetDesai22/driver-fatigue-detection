"""
Driver Fatigue Detection System — Desktop Dashboard
Pipeline: YOLO face detector -> MediaPipe face mesh (eye landmarks)
          -> CNN eye-state classifier (open/closed) -> drowsiness alarm
"""

import time
import math
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import torch
import torch.nn as nn
import numpy as np
import pygame
import mediapipe as mp
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image, ImageTk

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
FACE_MODEL_PATH = r"E:\driver-fatigue-detection\models\face_detector\train3\weights\best.pt"
EYE_MODEL_PATH  = r"E:\driver-fatigue-detection\models\eye_classifier\eye_model.pth"
ALARM_SOUND     = r"E:\driver-fatigue-detection\alarm.wav"
# EYES_CLOSED_ALERT_SECONDS is wall-clock time, not a frame count — a fixed frame threshold
# (the old CONSEC_FRAMES=15 approach) takes longer in real seconds whenever actual camera FPS
# drops (e.g. from the extra per-frame work added since: attention tracking, CARLA state sync,
# more sidebar widgets), which made the beep/meter feel like it was "taking too long" even
# though the frame count hadn't changed. Timing in seconds is predictable regardless of FPS,
# same approach already used for distraction detection and the CARLA fatigue escalation.
EYES_CLOSED_ALERT_SECONDS = 0.6
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEFT_EYE_LANDMARKS  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# ─────────────────────────────
# CARLA VEHICLE RESPONSE CONFIG (additive)
# A cumulative "strike" system: each SEPARATE drowsiness episode this session (i.e. each
# time the eyes stay closed for EYES_CLOSED_ALERT_SECONDS and the beep fires — that existing
# alarm logic is untouched) permanently escalates the vehicle's behavior, rather than only
# reacting for as long as the eyes happen to stay closed. The car does not trust a driver
# who's already been caught twice, even after they open their eyes again.
#   Strike 1 -> reduced speed
#   Strike 2 -> further reduced speed + move to the last (rightmost) lane
#   Strike 3+ -> full stop; stays stopped until a driver change is confirmed (automatically
#                by face recognition, or via the manual "Change Driver" button)
# ─────────────────────────────
FATIGUE_STRIKE_1_SPEED_KMH = 25.0
FATIGUE_STRIKE_2_SPEED_KMH = 12.0
FATIGUE_STOP_STRIKES = 3

# ─────────────────────────────
# DRIVER-CHANGE FACE RECOGNITION (additive)
# Uses the trained YOLO face detector (previously loaded but unused at inference — this is
# its first real use) to crop a face, and OpenCV's LBPH recognizer (opencv-contrib-python,
# already a dependency) to compare it against the current driver's reference captured during
# calibration. This is a lightweight heuristic, not production-grade face ID — LBPH distance
# thresholds are sensitive to lighting/angle and may need retuning for your camera.
# Only runs while the vehicle is stopped (3+ strikes) — zero cost during normal driving.
# ─────────────────────────────
DRIVER_CHANGE_LBPH_THRESHOLD = 75.0   # LBPH distance above this = "not the same person"
DRIVER_CHANGE_CONFIRM_COUNT = 3       # consecutive mismatched checks required (debounce)
DRIVER_CHECK_EVERY_N_FRAMES = 10      # throttle YOLO+LBPH checks while stopped
MIN_CALIB_FACE_SAMPLES = 3            # minimum reference crops needed to enable auto-detection
DRIVER_CHANGE_ANNOUNCE_SECONDS = 3.0

# Distraction (head turned away) is transient, not a strike — it only affects the vehicle
# while the driver is actively DISTRACTED and reverts the instant attention returns.
# Drowsiness strikes above always take priority over a mere distraction response.
DISTRACTION_SPEED_KMH = 28.0
DISTRACTION_STEER_BIAS = 0.12    # small, safe lane-weave while looking away

CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_CONNECT_TIMEOUT = 5.0      # seconds — fail fast to the local fallback sim if no server
CARLA_NORMAL_SPEED_KMH = 40.0
CARLA_HUD_PERIOD = 0.25          # seconds between HUD text redraws (see _draw_hud)

# ─────────────────────────────
# DISTRACTION DETECTION CONFIG
# (additive module — does not affect eye/drowsiness config above)
# ─────────────────────────────
DISTRACTION_YAW_THRESHOLD        = 12.0   # degrees, relative to calibrated baseline
DISTRACTION_PITCH_DOWN_THRESHOLD = 10.0   # looking down
DISTRACTION_PITCH_UP_THRESHOLD   = 10.0   # looking up
DISTRACTION_ROLL_THRESHOLD       = 18.0   # head tilt (secondary signal, not a primary trigger)
DISTRACTION_DURATION             = 1.5    # seconds of sustained deviation before confirmed DISTRACTED
# These were lowered from an initial 20/15/25 deg — the generic 6-point solvePnP model tends
# to underestimate real head rotation, so those defaults barely triggered on real head turns.
# If it's still not sensitive enough (or too twitchy) for your camera, watch the live Yaw/Pitch
# numbers in the sidebar while turning your head and retune these to match what you observe.
POSE_SMOOTHING_ALPHA             = 0.35   # EMA smoothing factor for yaw/pitch/roll
CALIBRATION_SECONDS              = 2.5    # baseline calibration duration at session start
# Flip these to -1 if "looking left/right/up/down" ever comes out mirrored on your camera —
# verify once with a live test: look right, confirm the sidebar shows "LOOKING RIGHT".
# PITCH_SIGN was determined analytically (see scratchpad test_attention.py) to need -1 for
# this 6-point solvePnP model: raw pitch increases when the nose/chin move DOWN in-frame,
# which is the opposite of the "positive pitch = looking up" convention detect_distraction_direction()
# uses. Still worth a 10-second live sanity check: look down, confirm the sidebar says "LOOKING DOWN".
YAW_SIGN   = 1
PITCH_SIGN = -1

# 6 stable MediaPipe Face Mesh landmarks used for solvePnP head-pose estimation:
# nose tip, chin, left-eye left corner, right-eye right corner, left mouth corner, right mouth corner
HEAD_POSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# Generic 3D face model points (mm) matching the landmarks above, standard for
# OpenCV solvePnP-based head-pose estimation.
HEAD_POSE_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),           # nose tip
    (0.0, -330.0, -65.0),      # chin
    (-225.0, 170.0, -135.0),   # left eye left corner
    (225.0, 170.0, -135.0),    # right eye right corner
    (-150.0, -150.0, -125.0),  # left mouth corner
    (150.0, -150.0, -125.0),   # right mouth corner
], dtype=np.float64)

# ─────────────────────────────
# THEME
# ─────────────────────────────
BG        = "#0f1117"
PANEL_BG  = "#171a24"
CARD_BG   = "#1e2230"
ACCENT    = "#00d3a7"
ALERT     = "#ff4d5e"
TEXT      = "#eef0f5"
SUBTEXT   = "#8b93a7"
FONT      = "Segoe UI"
SUBTEXT_BGR = (167, 147, 139)  # SUBTEXT hex, reordered for cv2 (BGR) overlay text


class EyeCNN(nn.Module):
    def __init__(self):
        super(EyeCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))


EYE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((24, 24)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def get_eye_roi(landmarks, indices, frame, padding=10):
    h, w = frame.shape[:2]
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x1 = max(0, min(p[0] for p in points) - padding)
    y1 = max(0, min(p[1] for p in points) - padding)
    x2 = min(w, max(p[0] for p in points) + padding)
    y2 = min(h, max(p[1] for p in points) + padding)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


# ─────────────────────────────
# DISTRACTION DETECTION — HEAD POSE (additive module)
# Reuses the same MediaPipe Face Mesh landmarks the eye-ROI code above already
# extracts; no additional model or dependency is introduced.
# ─────────────────────────────
def estimate_head_pose(landmarks, frame_width, frame_height):
    """Estimate (yaw, pitch, roll) in degrees from Face Mesh landmarks via solvePnP.
    Returns None if pose cannot be estimated for this frame."""
    try:
        image_points = np.array([
            (landmarks[i].x * frame_width, landmarks[i].y * frame_height)
            for i in HEAD_POSE_LANDMARK_IDS
        ], dtype=np.float64)

        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        ok, rotation_vec, _ = cv2.solvePnP(
            HEAD_POSE_MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
        if sy >= 1e-6:
            pitch = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
            yaw   = np.arctan2(-rotation_mat[2, 0], sy)
            roll  = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
        else:
            pitch = np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1])
            yaw   = np.arctan2(-rotation_mat[2, 0], sy)
            roll  = 0.0

        return {
            "yaw": float(np.degrees(yaw)),
            "pitch": float(np.degrees(pitch)),
            "roll": float(np.degrees(roll)),
        }
    except Exception:
        return None


def detect_distraction_direction(rel_yaw, rel_pitch, rel_roll):
    """Classify the approximate direction of a head-pose deviation. Yaw/pitch are
    the primary distraction signal; roll (tilt) is a secondary, lower-priority signal."""
    if rel_yaw <= -DISTRACTION_YAW_THRESHOLD:
        return "LOOKING LEFT"
    if rel_yaw >= DISTRACTION_YAW_THRESHOLD:
        return "LOOKING RIGHT"
    if rel_pitch <= -DISTRACTION_PITCH_DOWN_THRESHOLD:
        return "LOOKING DOWN"
    if rel_pitch >= DISTRACTION_PITCH_UP_THRESHOLD:
        return "LOOKING UP"
    if abs(rel_roll) >= DISTRACTION_ROLL_THRESHOLD:
        return "HEAD TILT"
    return "FORWARD"


def _circular_mean(angles_deg):
    """Mean of angles in degrees, computed via their unit-vector representation so it
    doesn't break near the +/-180 wraparound. This solvePnP model places the pitch
    axis for a frontal face close to that boundary, so a plain arithmetic mean is unsafe."""
    if not angles_deg:
        return 0.0
    sx = sum(math.cos(math.radians(a)) for a in angles_deg)
    sy = sum(math.sin(math.radians(a)) for a in angles_deg)
    return math.degrees(math.atan2(sy, sx))


def _ema_angle(prev_deg, new_deg, alpha):
    """Exponential moving average for a circular quantity (degrees), via unit vectors,
    so smoothing doesn't jump wildly when the raw value crosses +/-180."""
    px, py = math.cos(math.radians(prev_deg)), math.sin(math.radians(prev_deg))
    nx, ny = math.cos(math.radians(new_deg)), math.sin(math.radians(new_deg))
    x = alpha * nx + (1 - alpha) * px
    y = alpha * ny + (1 - alpha) * py
    return math.degrees(math.atan2(y, x))


def _angle_delta(current_deg, reference_deg):
    """Shortest signed circular difference (current - reference), wrapped to (-180, 180]."""
    return (current_deg - reference_deg + 180) % 360 - 180


def compute_baseline(pose_samples):
    """Average a list of {'yaw','pitch','roll'} samples collected during calibration
    into a driver-specific neutral baseline. Returns (0, 0, 0) if no samples were captured."""
    if not pose_samples:
        return 0.0, 0.0, 0.0
    yaws    = [p["yaw"] for p in pose_samples]
    pitches = [p["pitch"] for p in pose_samples]
    rolls   = [p["roll"] for p in pose_samples]
    return _circular_mean(yaws), _circular_mean(pitches), _circular_mean(rolls)


class AttentionTracker:
    """Tracks head-pose based driver attention (ATTENTIVE / DISTRACTED / UNKNOWN),
    independent of and additive to the existing eye-based drowsiness system.

    Uses a calibrated per-driver baseline, EMA smoothing, and a sustained-duration
    confirmation window (DISTRACTION_DURATION) so brief glances / natural movement /
    blinking never trigger a false distraction event.
    """

    def __init__(self):
        self.baseline_yaw = 0.0
        self.baseline_pitch = 0.0
        self.baseline_roll = 0.0
        self.calibrated = False

        self._smoothed_yaw = 0.0
        self._smoothed_pitch = 0.0
        self._smoothed_roll = 0.0
        self._has_smoothed = False

        self.state = "UNKNOWN"      # ATTENTIVE / DISTRACTED / UNKNOWN
        self.direction = "FORWARD"
        self.attention_score = 100.0
        self.distraction_duration = 0.0
        self.distraction_events = 0
        self.history = []           # completed distraction events (dicts)

        self._candidate_direction = None
        self._candidate_since = None
        self._active_event = None   # in-progress confirmed-distraction record

    def calibrate(self, baseline_yaw, baseline_pitch, baseline_roll):
        self.baseline_yaw = baseline_yaw
        self.baseline_pitch = baseline_pitch
        self.baseline_roll = baseline_roll
        self.calibrated = True

    def _reset_candidate(self):
        self._candidate_direction = None
        self._candidate_since = None

    def _close_active_event(self, now):
        if self._active_event is None:
            return
        ev = self._active_event
        self.history.append({
            "timestamp": ev["start_time"],
            "direction": ev["direction"],
            "duration": now - ev["start_time"],
            "max_yaw": ev["max_yaw"],
            "max_pitch": ev["max_pitch"],
        })
        self._active_event = None

    def _compute_score(self, rel_yaw, rel_pitch, now):
        dead_zone = 5.0  # degrees of natural movement that isn't penalized
        yaw_dev = max(0.0, abs(rel_yaw) - dead_zone)
        pitch_dev = max(0.0, abs(rel_pitch) - dead_zone)
        magnitude_penalty = min(60.0, (yaw_dev + pitch_dev) * 1.2)

        duration_penalty = 0.0
        if self._candidate_since is not None:
            candidate_duration = now - self._candidate_since
            duration_penalty = min(40.0, (candidate_duration / DISTRACTION_DURATION) * 40.0)

        return max(0.0, min(100.0, 100.0 - magnitude_penalty - duration_penalty))

    def update(self, raw_pose, face_visible, now=None):
        """raw_pose: {'yaw','pitch','roll'} in degrees (already sign-corrected), or None.
        Returns a status dict for the dashboard/overlay to render."""
        now = now if now is not None else time.time()

        if not face_visible or raw_pose is None:
            self._reset_candidate()
            self._close_active_event(now)
            self._has_smoothed = False
            self.state = "UNKNOWN"
            self.direction = "FORWARD"
            self.distraction_duration = 0.0
            return self._status()

        yaw, pitch, roll = raw_pose["yaw"], raw_pose["pitch"], raw_pose["roll"]

        if not self._has_smoothed:
            self._smoothed_yaw, self._smoothed_pitch, self._smoothed_roll = yaw, pitch, roll
            self._has_smoothed = True
        else:
            a = POSE_SMOOTHING_ALPHA
            self._smoothed_yaw   = _ema_angle(self._smoothed_yaw, yaw, a)
            self._smoothed_pitch = _ema_angle(self._smoothed_pitch, pitch, a)
            self._smoothed_roll  = _ema_angle(self._smoothed_roll, roll, a)

        rel_yaw   = _angle_delta(self._smoothed_yaw, self.baseline_yaw)
        rel_pitch = _angle_delta(self._smoothed_pitch, self.baseline_pitch)
        rel_roll  = _angle_delta(self._smoothed_roll, self.baseline_roll)

        direction = detect_distraction_direction(rel_yaw, rel_pitch, rel_roll)
        # `direction` (and the duration below) always reflect what the head pose is doing
        # RIGHT NOW — they are not gated behind confirmation. Only `state` (ATTENTIVE vs
        # DISTRACTED) is debounced by DISTRACTION_DURATION, so a quick glance is visibly
        # tracked in the dashboard/overlay immediately, while the official DISTRACTED label
        # (and the event counter) still waits for a sustained deviation as required.
        self.direction = direction

        if direction == "FORWARD":
            self._reset_candidate()
            self._close_active_event(now)
            self.state = "ATTENTIVE"
            self.distraction_duration = 0.0
        else:
            if self._candidate_direction != direction:
                self._close_active_event(now)  # direction changed mid-distraction
                self._candidate_direction = direction
                self._candidate_since = now
            candidate_duration = now - self._candidate_since
            self.distraction_duration = candidate_duration

            if candidate_duration >= DISTRACTION_DURATION:
                if self.state != "DISTRACTED":
                    self.distraction_events += 1
                    self._active_event = {
                        "start_time": self._candidate_since,
                        "direction": direction,
                        "max_yaw": rel_yaw,
                        "max_pitch": rel_pitch,
                    }
                else:
                    if abs(rel_yaw) > abs(self._active_event["max_yaw"]):
                        self._active_event["max_yaw"] = rel_yaw
                    if abs(rel_pitch) > abs(self._active_event["max_pitch"]):
                        self._active_event["max_pitch"] = rel_pitch
                self.state = "DISTRACTED"
            else:
                # brief glance / still confirming — not yet counted as a distraction event.
                # distraction_duration is left at candidate_duration (set above) so the
                # dashboard can show a live "building up" timer instead of freezing at 0.
                self.state = "ATTENTIVE"

        self.attention_score = self._compute_score(rel_yaw, rel_pitch, now)
        return self._status(rel_yaw, rel_pitch, rel_roll)

    def _status(self, rel_yaw=0.0, rel_pitch=0.0, rel_roll=0.0):
        return {
            "state": self.state,
            "direction": self.direction,
            "yaw": rel_yaw,
            "pitch": rel_pitch,
            "roll": rel_roll,
            "duration": self.distraction_duration,
            "score": self.attention_score,
            "events": self.distraction_events,
        }

    def get_session_summary(self, session_duration):
        """Aggregate stats for the whole session — shown when the session ends."""
        events = list(self.history)
        if self._active_event is not None:
            now = time.time()
            events = events + [{
                "timestamp": self._active_event["start_time"],
                "direction": self._active_event["direction"],
                "duration": now - self._active_event["start_time"],
                "max_yaw": self._active_event["max_yaw"],
                "max_pitch": self._active_event["max_pitch"],
            }]

        total_distracted = sum(e["duration"] for e in events)
        longest = max((e["duration"] for e in events), default=0.0)
        avg = (total_distracted / len(events)) if events else 0.0

        direction_counts = {}
        for e in events:
            direction_counts[e["direction"]] = direction_counts.get(e["direction"], 0) + 1
        most_common = max(direction_counts, key=direction_counts.get) if direction_counts else "—"

        attentive_time = max(0.0, session_duration - total_distracted)
        attention_rate = (attentive_time / session_duration * 100.0) if session_duration > 0 else 100.0

        return {
            "session_duration": session_duration,
            "attentive_time": attentive_time,
            "distracted_time": total_distracted,
            "events": len(events),
            "longest_event": longest,
            "avg_event_duration": avg,
            "most_common_direction": most_common,
            "attention_rate": attention_rate,
        }


class VehicleSimulation:
    """A lightweight, self-contained 'digital twin' visualization — a simulated vehicle
    reacting live to the same Driver State (fatigue + attention) the dashboard already
    computes every frame. Runs in its own Toplevel window so it can sit side-by-side with
    the detection window during a demo.

    This is purely a consumer of existing state passed in via update() — it does not touch
    the eye/CNN drowsiness pipeline, the AttentionTracker, or any detection logic at all.
    It exists as a zero-install stand-in for a real vehicle-control integration (e.g. CARLA
    or a real CAN bus), showing the same response logic documented in the project report.
    """

    CANVAS_W = 420
    CANVAS_H = 560
    ROAD_W = 260

    def __init__(self, root):
        self.closed = False
        self.top = tk.Toplevel(root)
        self.top.title("Vehicle Simulation — Live Driver State")
        self.top.configure(bg=BG)
        self.top.geometry(f"{self.CANVAS_W + 40}x{self.CANVAS_H + 90}")
        self.top.protocol("WM_DELETE_WINDOW", self.close)

        self.lane_center_x = self.CANVAS_W // 2

        tk.Label(self.top, text="🚗  VEHICLE SIMULATION", font=(FONT, 13, "bold"),
                 fg=TEXT, bg=BG).pack(pady=(10, 2))
        tk.Label(self.top, text="A simulated vehicle reacting live to the driver state above",
                 font=(FONT, 8), fg=SUBTEXT, bg=BG).pack(pady=(0, 6))
        self.status_label = tk.Label(self.top, text="AUTOPILOT: NORMAL", font=(FONT, 11, "bold"),
                                      fg=ACCENT, bg=BG)
        self.status_label.pack(pady=(0, 2))
        self.speed_label = tk.Label(self.top, text="Speed: 0 km/h  |  Strikes: 0",
                                     font=(FONT, 9), fg=SUBTEXT, bg=BG)
        self.speed_label.pack(pady=(0, 8))

        self.canvas = tk.Canvas(self.top, width=self.CANVAS_W, height=self.CANVAS_H,
                                 bg="#2b2f3a", highlightthickness=0)
        self.canvas.pack(padx=20, pady=(0, 10))

        # animation state
        self.lane_offset = 0.0
        self.speed = 6.0          # px/frame scroll speed — represents vehicle speed
        self.car_x_offset = 0.0   # lateral offset from lane center (px)
        self.target_offset = 0.0
        self.hazard_on = False
        self._hazard_tick = 0
        self._announcement = None
        self._announcement_until = 0.0

        self._draw_road()
        self._init_dynamic_shapes()

    def announce(self, message, duration=DRIVER_CHANGE_ANNOUNCE_SECONDS):
        self._announcement = message
        self._announcement_until = time.time() + duration

    def _draw_road(self):
        c = self.canvas
        road_x1 = self.lane_center_x - self.ROAD_W // 2
        road_x2 = self.lane_center_x + self.ROAD_W // 2
        c.create_rectangle(road_x1, 0, road_x2, self.CANVAS_H, fill="#3a3f4a", outline="")
        c.create_rectangle(road_x2, 0, road_x2 + 30, self.CANVAS_H, fill="#4a4030", outline="")  # shoulder

    def _init_dynamic_shapes(self):
        c = self.canvas
        self.lane_marks = []
        for i in range(-1, self.CANVAS_H // 40 + 2):
            y = i * 40
            seg = c.create_rectangle(self.lane_center_x - 3, y, self.lane_center_x + 3, y + 20,
                                      fill="#d8d8d8", outline="")
            self.lane_marks.append(seg)

        self.car_y = self.CANVAS_H - 130
        self.car_body = c.create_rectangle(
            self.lane_center_x - 22, self.car_y, self.lane_center_x + 22, self.car_y + 60,
            fill=ACCENT, outline="#0f1117", width=2)
        self.hazard_l = c.create_oval(0, 0, 8, 8, fill="#333", outline="")
        self.hazard_r = c.create_oval(0, 0, 8, 8, fill="#333", outline="")

    def close(self):
        self.closed = True
        try:
            self.top.destroy()
        except Exception:
            pass

    def update(self, strikes, attention_state, direction):
        """Called once per detection frame with the live driver state. `strikes` mirrors
        CarlaBridge exactly: a cumulative, persistent count of drowsiness episodes this
        session — behavior escalates and STAYS escalated, it does not reset just because
        the eyes reopen. 3+ strikes stops the vehicle until strikes are reset externally
        (Change Driver). Cheap canvas coordinate updates only — negligible cost."""
        if self.closed:
            return
        try:
            if strikes >= 3:
                self.target_offset = 85.0  # parked on the shoulder
                target_speed = 0.0
                status_text, color = "STOPPED — CHANGE DRIVER TO RESUME", ALERT
                self.hazard_on = True
            elif strikes == 2:
                self.target_offset = 70.0  # merged into the last lane
                target_speed = 2.0
                status_text, color = "SLOWING DOWN — LAST LANE (2nd strike)", "#ffb703"
                self.hazard_on = False
            elif strikes == 1:
                self.target_offset = 0.0
                target_speed = 4.0
                status_text, color = "SLOWING DOWN (1st strike)", "#ffb703"
                self.hazard_on = False
            elif attention_state == "DISTRACTED":
                if direction == "LOOKING RIGHT":
                    self.target_offset = 55.0
                elif direction == "LOOKING LEFT":
                    self.target_offset = -55.0
                else:
                    self.target_offset = 0.0
                target_speed = 5.0
                status_text, color = f"LANE DRIFT WARNING — {direction}", "#ffb703"
                self.hazard_on = False
            else:
                self.target_offset = 0.0
                target_speed = 6.0
                status_text, color = "AUTOPILOT: NORMAL", ACCENT
                self.hazard_on = False

            if time.time() < self._announcement_until and self._announcement:
                status_text, color = self._announcement, ACCENT

            self.status_label.config(text=status_text, fg=color)
            self.speed_label.config(text=f"Speed: {self.speed * 6:.0f} km/h   |   Strikes: {strikes}")

            self.car_x_offset += (self.target_offset - self.car_x_offset) * 0.08
            self.speed += (target_speed - self.speed) * 0.06

            self.lane_offset = (self.lane_offset + self.speed) % 40
            for i, seg in enumerate(self.lane_marks):
                y = i * 40 - 40 + self.lane_offset
                self.canvas.coords(seg, self.lane_center_x - 3, y, self.lane_center_x + 3, y + 20)

            cx = self.lane_center_x + self.car_x_offset
            self.canvas.coords(self.car_body, cx - 22, self.car_y, cx + 22, self.car_y + 60)
            self.canvas.coords(self.hazard_l, cx - 20, self.car_y + 4, cx - 12, self.car_y + 12)
            self.canvas.coords(self.hazard_r, cx + 12, self.car_y + 4, cx + 20, self.car_y + 12)

            self._hazard_tick += 1
            if self.hazard_on:
                blink = "#ff9900" if (self._hazard_tick // 8) % 2 == 0 else "#333"
                self.canvas.itemconfig(self.hazard_l, fill=blink)
                self.canvas.itemconfig(self.hazard_r, fill=blink)
            else:
                self.canvas.itemconfig(self.hazard_l, fill="#333")
                self.canvas.itemconfig(self.hazard_r, fill="#333")
        except tk.TclError:
            self.closed = True


class CarlaBridge:
    """Connects to a running CARLA simulator (the CarlaUE4/CarlaUnreal server must already
    be running separately — this only starts the Python client side) and manually drives a
    spawned vehicle, throttling/braking/steering it based on the driver's cumulative
    "strike" count computed by the detection loop:

        0 strikes — normal speed, current lane
        1 strike  — reduced speed (the beep itself is the existing pygame alarm, unchanged)
        2 strikes — further reduced speed + merges into the last (rightmost) lane
        3+ strikes — full stop; stays stopped until the app resets strikes (Change Driver)

    All status messages and speed are rendered directly inside the CARLA world as debug
    HUD text above the vehicle — nothing about the vehicle is shown in the Tkinter app.

    Runs entirely in a background thread so it never blocks the Tkinter UI or the detection
    loop. If the `carla` package is missing or no server is reachable, it fails fast (see
    CARLA_CONNECT_TIMEOUT) and reports that via `.error`/`.status_text` so the caller can fall
    back to the local Tkinter VehicleSimulation instead.
    """

    def __init__(self, host=CARLA_HOST, port=CARLA_PORT):
        self.host = host
        self.port = port
        self.connected = False
        self.error = None
        self.running = False
        self.strikes = 0
        self.attention_state = "ATTENTIVE"
        self.direction = "FORWARD"
        self.status_text = "Not started"
        self.vehicle = None
        self.world_map = None
        self._thread = None
        self._hud_tick = 0
        self.announcement = None
        self.announcement_until = 0.0

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def set_driver_state(self, strikes, attention_state, direction):
        self.strikes = strikes
        self.attention_state = attention_state
        self.direction = direction

    def announce(self, message, duration=DRIVER_CHANGE_ANNOUNCE_SECONDS):
        """Briefly overrides the HUD with a one-off message (e.g. 'DRIVER CHANGED') that
        fades back to the normal status text after `duration` seconds."""
        self.announcement = message
        self.announcement_until = time.time() + duration

    def _run(self):
        try:
            import carla
        except ImportError:
            self.error = "carla package not installed in this venv (pip install carla==0.9.16)"
            self.status_text = self.error
            return

        try:
            client = carla.Client(self.host, self.port)
            client.set_timeout(CARLA_CONNECT_TIMEOUT)
            world = client.get_world()
            self.world_map = world.get_map()

            blueprint_library = world.get_blueprint_library()
            candidates = blueprint_library.filter("vehicle.tesla.model3")
            vehicle_bp = candidates[0] if candidates else blueprint_library.filter("vehicle.*")[0]

            spawn_points = self.world_map.get_spawn_points()
            if not spawn_points:
                self.error = "No spawn points on the current CARLA map"
                self.status_text = self.error
                return

            for sp in spawn_points[:15]:
                self.vehicle = world.try_spawn_actor(vehicle_bp, sp)
                if self.vehicle is not None:
                    break
            if self.vehicle is None:
                self.error = "Could not spawn vehicle (tried 15 spawn points, all occupied)"
                self.status_text = self.error
                return

            spectator = world.get_spectator()
            self.connected = True
            self.status_text = "Connected — driving"

            while self.running:
                if not self.vehicle.is_alive:
                    break

                transform = self.vehicle.get_transform()
                velocity = self.vehicle.get_velocity()
                speed_kmh = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

                strikes = self.strikes
                attention_state = self.attention_state
                direction = self.direction
                use_last_lane = strikes >= 2
                drift_bias = 0.0

                # Drowsiness (eye closure) always takes priority over mere distraction —
                # a driver who's already been caught drowsy is the higher risk, even if
                # they're currently looking forward.
                if strikes >= FATIGUE_STOP_STRIKES:
                    target_speed = 0.0
                    hud_msg = "DRIVER UNRESPONSIVE — VEHICLE STOPPED (change driver to resume)"
                    hud_color = carla.Color(220, 40, 40)
                elif strikes == 2:
                    target_speed = FATIGUE_STRIKE_2_SPEED_KMH
                    hud_msg = "DROWSINESS DETECTED (2nd time) — slowing down & moving to last lane"
                    hud_color = carla.Color(230, 140, 0)
                elif strikes == 1:
                    target_speed = FATIGUE_STRIKE_1_SPEED_KMH
                    hud_msg = "DROWSINESS DETECTED (1st time) — slowing down"
                    hud_color = carla.Color(230, 200, 0)
                elif attention_state == "DISTRACTED":
                    # Transient response — reverts the instant attention returns to forward,
                    # unlike the persistent drowsiness strikes above.
                    target_speed = DISTRACTION_SPEED_KMH
                    hud_msg = f"DISTRACTED — {direction} — slowing down"
                    hud_color = carla.Color(60, 140, 230)
                    if direction == "LOOKING RIGHT":
                        drift_bias = DISTRACTION_STEER_BIAS
                    elif direction == "LOOKING LEFT":
                        drift_bias = -DISTRACTION_STEER_BIAS
                else:
                    target_speed = CARLA_NORMAL_SPEED_KMH
                    hud_msg = "AUTOPILOT: NORMAL"
                    hud_color = carla.Color(0, 200, 120)

                if time.time() < self.announcement_until and self.announcement:
                    hud_msg = self.announcement
                    hud_color = carla.Color(0, 220, 140)

                steer = self._compute_steer(carla, transform, use_last_lane, drift_bias)

                if target_speed <= 0.1:
                    control = carla.VehicleControl(
                        throttle=0.0, steer=steer, brake=1.0, hand_brake=(speed_kmh < 1.0))
                else:
                    diff = target_speed - speed_kmh
                    if diff > 0:
                        control = carla.VehicleControl(
                            throttle=max(0.0, min(0.75, diff * 0.05)), steer=steer, brake=0.0)
                    else:
                        control = carla.VehicleControl(
                            throttle=0.0, steer=steer, brake=max(0.0, min(0.6, -diff * 0.05)))

                self.vehicle.apply_control(control)
                self._update_chase_camera(carla, spectator, transform)

                self._hud_tick += 1
                if self._hud_tick % 5 == 0:  # ~4 Hz — matches CARLA_HUD_PERIOD, no stacking
                    self._draw_hud(carla, world, transform, hud_msg, hud_color, speed_kmh, strikes)

                self.status_text = f"Strikes {strikes} — {speed_kmh:.0f} km/h (target {target_speed:.0f})"
                time.sleep(0.05)  # ~20 Hz control loop

        except Exception as exc:
            self.error = str(exc)
            self.status_text = f"CARLA error: {exc}"
        finally:
            self._cleanup()

    def _compute_steer(self, carla, transform, use_last_lane, drift_bias=0.0):
        """Waypoint-following proportional steering so the vehicle stays on the road.
        When use_last_lane is True, walks right lane-by-lane to find the rightmost
        drivable lane and steers toward that instead of the current one — this is what
        makes the vehicle merge over after the 2nd drowsiness strike. drift_bias adds a
        small, bounded offset on top (used to visualize distraction as a gentle lane
        weave without risking the vehicle leaving the road, since the waypoint-follow
        term still dominates)."""
        try:
            location = transform.location
            waypoint = self.world_map.get_waypoint(location)
            if use_last_lane:
                candidate = waypoint
                for _ in range(6):  # walk right at most a few lanes
                    right = candidate.get_right_lane()
                    if right is None or right.lane_type != carla.LaneType.Driving:
                        break
                    candidate = right
                waypoint = candidate
            next_wps = waypoint.next(5.0)
            if not next_wps:
                return max(-1.0, min(1.0, drift_bias))
            target = next_wps[0].transform.location
            dx, dy = target.x - location.x, target.y - location.y
            yaw = math.radians(transform.rotation.yaw)
            target_angle = math.atan2(dy, dx)
            angle_diff = (target_angle - yaw + math.pi) % (2 * math.pi) - math.pi
            base_steer = angle_diff * 1.2
            return max(-1.0, min(1.0, base_steer + drift_bias))
        except Exception:
            return 0.0

    def _draw_hud(self, carla, world, transform, message, color, speed_kmh, strikes):
        """Draws the status message and speed as floating text above the vehicle inside
        the CARLA world itself — this is the ONLY place vehicle status is shown. Called
        at a reduced, fixed cadence (see CARLA_HUD_PERIOD) with a matching life_time so
        the previous text has just finished fading before the next one is drawn —
        drawing this every control-loop tick instead would leave 2-3 overlapping copies
        visible at once as the vehicle moves between calls."""
        try:
            base = transform.location
            life = CARLA_HUD_PERIOD + 0.02
            world.debug.draw_string(base + carla.Location(z=3.4), message, draw_shadow=True,
                                     color=color, life_time=life, persistent_lines=False)
            world.debug.draw_string(
                base + carla.Location(z=2.8),
                f"Speed: {speed_kmh:.0f} km/h   |   Strikes: {strikes}",
                draw_shadow=True, color=color, life_time=life, persistent_lines=False)
        except Exception:
            pass

    def _update_chase_camera(self, carla, spectator, transform):
        try:
            cam_location = transform.location + carla.Location(z=12) - transform.get_forward_vector() * 10
            spectator.set_transform(carla.Transform(
                cam_location, carla.Rotation(pitch=-35, yaw=transform.rotation.yaw)))
        except Exception:
            pass

    def _cleanup(self):
        try:
            if self.vehicle is not None:
                self.vehicle.destroy()
        except Exception:
            pass
        self.connected = False
        self.vehicle = None


class FatigueApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Fatigue Detection System")
        self.root.geometry("1200x700")
        self.root.minsize(1040, 640)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.show_face_mesh_var = tk.BooleanVar(value=True)

        # runtime state
        self.cap = None
        self.running = False
        self.face_model = None
        self.eye_model = None
        self.face_mesh = None
        self.models_ready = False
        self.alarm_playing = False

        self.eyes_closed_since = None
        self.blink_count = 0
        self.alert_count = 0
        self.alarm_active = False
        self._was_closed_seq = False
        self.session_start = None
        self._prev_frame_time = time.time()
        self.fps = 0.0

        # ── distraction detection state (additive) ──
        self.attention_tracker = AttentionTracker()
        self.calibrating = False
        self.calib_samples = []
        self.calib_start_time = None

        # ── vehicle simulation (additive) ──
        # CarlaBridge is the real simulator; VehicleSimulation (sim_window) is a zero-install
        # local fallback used automatically if no CARLA server is reachable. fatigue_strikes
        # is cumulative and persistent across the session (see FATIGUE_STRIKE_* config).
        self.sim_window = None
        self.carla_bridge = None
        self._carla_fallback_started = False
        self.fatigue_strikes = 0

        # ── driver-change face recognition (additive) ──
        self.calib_face_samples = []
        self.driver_recognizer = None
        self.driver_recognizer_ready = False
        self._driver_mismatch_count = 0
        self._driver_check_tick = 0
        self._last_face_capture_time = 0.0

        self._build_start_screen()
        threading.Thread(target=self._load_models, daemon=True).start()

    # ── STARTUP / MODEL LOADING ──────────────────────────
    def _load_models(self):
        try:
            pygame.mixer.init()
            self.face_model = YOLO(FACE_MODEL_PATH)
            self.eye_model = EyeCNN().to(DEVICE)
            self.eye_model.load_state_dict(torch.load(EYE_MODEL_PATH, map_location=DEVICE))
            self.eye_model.eval()
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.models_ready = True
            self.root.after(0, self._on_models_ready)
        except Exception as exc:
            self.root.after(0, lambda: self._on_models_failed(exc))

    def _on_models_ready(self):
        self.status_line.config(text=f"Ready  ·  running on {DEVICE.type.upper()}", fg=ACCENT)
        self.start_btn.config(state="normal", bg=ACCENT)

    def _on_models_failed(self, exc):
        self.status_line.config(text=f"Failed to load models: {exc}", fg=ALERT)

    # ── START SCREEN ──────────────────────────────────────
    def _build_start_screen(self):
        self.start_frame = tk.Frame(self.root, bg=BG)
        self.start_frame.pack(fill="both", expand=True)

        wrap = tk.Frame(self.start_frame, bg=BG)
        wrap.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(wrap, text="🚗  DRIVER FATIGUE DETECTION SYSTEM",
                 font=(FONT, 26, "bold"), fg=TEXT, bg=BG).pack(pady=(0, 6))
        tk.Label(wrap, text="Real-time drowsiness monitoring using YOLO + MediaPipe + CNN",
                 font=(FONT, 12), fg=SUBTEXT, bg=BG).pack(pady=(0, 30))

        pipeline = tk.Frame(wrap, bg=BG)
        pipeline.pack(pady=(0, 30))
        steps = [
            ("1", "Face Detection", "YOLO"),
            ("2", "Eye Landmarks", "MediaPipe Face Mesh"),
            ("3", "Eye-State Classification", "Custom CNN"),
            ("4", "Drowsiness Alert", "Audio Alarm"),
        ]
        for i, (num, title, sub) in enumerate(steps):
            col = i * 2
            card = tk.Frame(pipeline, bg=CARD_BG, width=170, height=100)
            card.grid(row=0, column=col, padx=6)
            card.grid_propagate(False)
            tk.Label(card, text=num, font=(FONT, 14, "bold"), fg=ACCENT, bg=CARD_BG).pack(pady=(14, 2))
            tk.Label(card, text=title, font=(FONT, 10, "bold"), fg=TEXT, bg=CARD_BG,
                     wraplength=150, justify="center").pack()
            tk.Label(card, text=sub, font=(FONT, 9), fg=SUBTEXT, bg=CARD_BG).pack(pady=(2, 0))
            if i < len(steps) - 1:
                tk.Label(pipeline, text="→", font=(FONT, 16), fg=SUBTEXT, bg=BG).grid(row=0, column=col + 1)

        self.start_btn = tk.Button(
            wrap, text="▶  START DETECTION", font=(FONT, 13, "bold"),
            fg="#0f1117", bg="#3a3f52", activebackground=ACCENT, bd=0,
            padx=30, pady=12, cursor="hand2", state="disabled",
            command=self._start_detection,
        )
        self.start_btn.pack(pady=(0, 10))

        self.status_line = tk.Label(wrap, text="Loading AI models…", font=(FONT, 10), fg=SUBTEXT, bg=BG)
        self.status_line.pack()

    # ── DETECTION SCREEN ──────────────────────────────────
    def _build_detection_screen(self):
        self.det_frame = tk.Frame(self.root, bg=BG)

        video_wrap = tk.Frame(self.det_frame, bg=BG)
        video_wrap.pack(side="left", fill="both", expand=True, padx=(16, 8), pady=16)

        self.video_label = tk.Label(video_wrap, bg="#000000")
        self.video_label.pack(fill="both", expand=True)

        sidebar_container = tk.Frame(self.det_frame, bg=PANEL_BG, width=300)
        sidebar_container.pack(side="right", fill="y", padx=(8, 16), pady=16)
        sidebar_container.pack_propagate(False)

        sidebar_canvas = tk.Canvas(sidebar_container, bg=PANEL_BG, highlightthickness=0)
        sidebar_scroll = ttk.Scrollbar(sidebar_container, orient="vertical", command=sidebar_canvas.yview)
        sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        sidebar_canvas.pack(side="left", fill="both", expand=True)
        sidebar_scroll.pack(side="right", fill="y")

        # All sidebar content lives in this inner frame, which scrolls if it grows
        # taller than the visible window — keeps the dashboard usable on smaller
        # displays (e.g. a projector) without clipping the Stop button.
        sidebar = tk.Frame(sidebar_canvas, bg=PANEL_BG)
        sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")

        def _on_sidebar_configure(_event=None):
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))

        def _on_canvas_configure(event):
            sidebar_canvas.itemconfig(sidebar_window, width=event.width)

        sidebar.bind("<Configure>", _on_sidebar_configure)
        sidebar_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            sidebar_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        sidebar_canvas.bind("<MouseWheel>", _on_mousewheel)
        sidebar.bind("<MouseWheel>", _on_mousewheel)

        mesh_row = tk.Frame(sidebar, bg=PANEL_BG)
        mesh_row.pack(fill="x", padx=20, pady=(18, 4))
        tk.Checkbutton(
            mesh_row, text="Show face mesh dots (468 landmarks)", variable=self.show_face_mesh_var,
            font=(FONT, 9), fg=TEXT, bg=PANEL_BG, activebackground=PANEL_BG,
            selectcolor=CARD_BG, anchor="w",
        ).pack(fill="x")

        tk.Label(sidebar, text="LIVE STATUS", font=(FONT, 10, "bold"), fg=SUBTEXT, bg=PANEL_BG).pack(
            anchor="w", padx=20, pady=(20, 6))

        self.status_badge = tk.Label(sidebar, text="AWAKE", font=(FONT, 20, "bold"),
                                      fg="#0f1117", bg=ACCENT, padx=10, pady=10)
        self.status_badge.pack(fill="x", padx=20, pady=(0, 20))

        self._sep(sidebar)

        self.eye_l_val = self._metric_row(sidebar, "Left Eye")
        self.eye_r_val = self._metric_row(sidebar, "Right Eye")

        self._sep(sidebar)

        tk.Label(sidebar, text="Drowsiness Meter", font=(FONT, 9, "bold"), fg=SUBTEXT, bg=PANEL_BG).pack(
            anchor="w", padx=20, pady=(10, 4))
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("meter.Horizontal.TProgressbar", troughcolor=CARD_BG, background=ACCENT, thickness=14)
        self.meter = ttk.Progressbar(sidebar, style="meter.Horizontal.TProgressbar",
                                      maximum=EYES_CLOSED_ALERT_SECONDS, length=260)
        self.meter.pack(padx=20, pady=(0, 20))

        self._sep(sidebar)

        self.session_val   = self._metric_row(sidebar, "Session Time")
        self.blink_val     = self._metric_row(sidebar, "Blinks")
        self.alert_val     = self._metric_row(sidebar, "Alerts Triggered")
        self.fps_val       = self._metric_row(sidebar, "FPS")

        # ── DRIVER ATTENTION (additive section) ──
        self._sep(sidebar)
        tk.Label(sidebar, text="DRIVER ATTENTION", font=(FONT, 10, "bold"), fg=SUBTEXT, bg=PANEL_BG).pack(
            anchor="w", padx=20, pady=(10, 6))

        self.attn_badge = tk.Label(sidebar, text="⚪ CALIBRATING…", font=(FONT, 14, "bold"),
                                    fg="#0f1117", bg=SUBTEXT, padx=10, pady=8)
        self.attn_badge.pack(fill="x", padx=20, pady=(0, 10))

        indicator_wrap = tk.Frame(sidebar, bg=PANEL_BG)
        indicator_wrap.pack(pady=(0, 8))
        self.direction_canvas = tk.Canvas(indicator_wrap, width=90, height=90, bg=CARD_BG, highlightthickness=0)
        self.direction_canvas.pack()
        self._draw_direction_base()
        self.direction_dot = self.direction_canvas.create_oval(40, 40, 50, 50, fill=ACCENT, outline="")

        self.direction_val         = self._metric_row(sidebar, "Direction")
        self.yaw_val               = self._metric_row(sidebar, "Yaw")
        self.pitch_val             = self._metric_row(sidebar, "Pitch")
        self.roll_val              = self._metric_row(sidebar, "Roll")
        self.distraction_dur_val   = self._metric_row(sidebar, "Distraction Duration")
        self.attn_score_val        = self._metric_row(sidebar, "Attention Score")
        self.distraction_events_val = self._metric_row(sidebar, "Distraction Events")

        # Vehicle status/speed messages are intentionally NOT shown here — they're drawn
        # live inside the CARLA world itself (or the fallback simulation window), never
        # on this dashboard. Only the reset control lives here.
        btn_wrap = tk.Frame(sidebar, bg=PANEL_BG)
        btn_wrap.pack(side="bottom", fill="x", padx=20, pady=20)
        tk.Button(btn_wrap, text="🔄  CHANGE DRIVER", font=(FONT, 10, "bold"),
                  fg=TEXT, bg=CARD_BG, activebackground="#ffb703", bd=0, pady=9,
                  cursor="hand2", command=self._change_driver).pack(fill="x", pady=(0, 8))
        tk.Button(btn_wrap, text="■  STOP & BACK", font=(FONT, 11, "bold"),
                  fg=TEXT, bg=CARD_BG, activebackground=ALERT, bd=0, pady=10,
                  cursor="hand2", command=self._stop_detection).pack(fill="x")

    def _sep(self, parent):
        tk.Frame(parent, bg=CARD_BG, height=1).pack(fill="x", padx=20, pady=6)

    def _metric_row(self, parent, label):
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill="x", padx=20, pady=4)
        tk.Label(row, text=label, font=(FONT, 10), fg=SUBTEXT, bg=PANEL_BG).pack(side="left")
        val = tk.Label(row, text="—", font=(FONT, 10, "bold"), fg=TEXT, bg=PANEL_BG)
        val.pack(side="right")
        return val

    # ── DISTRACTION DETECTION — UI helpers (additive) ──────
    def _draw_direction_base(self):
        c = self.direction_canvas
        c.create_oval(10, 10, 80, 80, outline=SUBTEXT)
        c.create_line(45, 10, 45, 80, fill=CARD_BG, width=1)
        c.create_line(10, 45, 80, 45, fill=CARD_BG, width=1)
        c.create_text(45, 6, text="UP", fill=SUBTEXT, font=(FONT, 6))
        c.create_text(45, 87, text="DOWN", fill=SUBTEXT, font=(FONT, 6))
        c.create_text(4, 45, text="L", fill=SUBTEXT, font=(FONT, 7))
        c.create_text(86, 45, text="R", fill=SUBTEXT, font=(FONT, 7))

    def _update_direction_indicator(self, yaw, pitch):
        radius = 30
        max_angle = 35.0
        dx = max(-1.0, min(1.0, yaw / max_angle)) * radius
        dy = max(-1.0, min(1.0, -pitch / max_angle)) * radius
        cx, cy = 45, 45
        self.direction_canvas.coords(self.direction_dot, cx + dx - 5, cy + dy - 5, cx + dx + 5, cy + dy + 5)

    def _get_signed_pose(self, landmarks, w, h):
        pose = estimate_head_pose(landmarks, w, h)
        if pose is None:
            return None
        return {
            "yaw": YAW_SIGN * pose["yaw"],
            "pitch": PITCH_SIGN * pose["pitch"],
            "roll": pose["roll"],
        }

    def _draw_attention_overlay(self, display, status):
        if status["state"] == "CALIBRATING":
            cv2.putText(display, "Calibrating... look straight at the camera",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 211, 167), 2)
            return
        if status["state"] == "UNKNOWN":
            cv2.putText(display, "ATTENTION: UNKNOWN (no face)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
            return
        watching = status["state"] == "ATTENTIVE" and status["direction"] != "FORWARD"
        if status["state"] == "DISTRACTED":
            color = (0, 210, 255)
            line = f"ATTENTION: DISTRACTED  {status['direction']}  {status['duration']:.1f}s"
        elif watching:
            color = (0, 165, 255)
            line = f"ATTENTION: watching {status['direction']}  {status['duration']:.1f}s"
        else:
            color = (0, 211, 167)
            line = "ATTENTION: ATTENTIVE"
        cv2.putText(display, line, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(display, f"Yaw {status['yaw']:+.1f}  Pitch {status['pitch']:+.1f}  Roll {status['roll']:+.1f}",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, SUBTEXT_BGR, 1)

    def _update_attention_sidebar(self, status):
        state = status["state"]
        if state == "CALIBRATING":
            self.attn_badge.config(text="⚪ CALIBRATING…", bg=SUBTEXT, fg="#0f1117")
        elif state == "UNKNOWN":
            self.attn_badge.config(text="⚫ UNKNOWN", bg="#3a3f52", fg=TEXT)
        elif state == "DISTRACTED":
            self.attn_badge.config(text="🟡 DISTRACTED", bg="#ffb703", fg="#1a1300")
        elif status["direction"] != "FORWARD":
            self.attn_badge.config(text="🟠 WATCHING…", bg="#ff8c42", fg="#1a1300")
        else:
            self.attn_badge.config(text="🟢 ATTENTIVE", bg=ACCENT, fg="#0f1117")

        self.direction_val.config(text=status["direction"])
        self.yaw_val.config(text=f"{status['yaw']:+.1f}°")
        self.pitch_val.config(text=f"{status['pitch']:+.1f}°")
        self.roll_val.config(text=f"{status['roll']:+.1f}°")
        self.distraction_dur_val.config(text=f"{status['duration']:.1f}s")
        self.attn_score_val.config(
            text=f"{status['score']:.0f}%",
            fg=ALERT if status["score"] < 50 else (ACCENT if status["score"] > 80 else "#ffb703"))
        self.distraction_events_val.config(text=str(status["events"]))

        if state not in ("UNKNOWN", "CALIBRATING"):
            self._update_direction_indicator(status["yaw"], status["pitch"])

    def _show_session_summary(self):
        duration = time.time() - self.session_start
        summary = self.attention_tracker.get_session_summary(duration)
        mm, ss = divmod(int(duration), 60)
        dm, ds = divmod(int(summary["distracted_time"]), 60)
        am, asec = divmod(int(summary["attentive_time"]), 60)
        text = (
            f"Session Duration: {mm:02d}:{ss:02d}\n\n"
            f"Attentive Time: {am:02d}:{asec:02d}\n"
            f"Distracted Time: {dm:02d}:{ds:02d}\n"
            f"Distraction Events: {summary['events']}\n"
            f"Longest Event: {summary['longest_event']:.1f}s\n"
            f"Avg Event Duration: {summary['avg_event_duration']:.1f}s\n"
            f"Most Common Direction: {summary['most_common_direction']}\n"
            f"Attention Rate: {summary['attention_rate']:.1f}%\n\n"
            f"Drowsiness Alerts Triggered: {self.alert_count}\n"
            f"Blinks Detected: {self.blink_count}"
        )
        messagebox.showinfo("Driver Attention Summary", text, parent=self.root)

    # ── FLOW CONTROL ──────────────────────────────────────
    def _start_detection(self):
        if not self.models_ready:
            return
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            self.status_line.config(text="Could not access webcam (index 0).", fg=ALERT)
            return

        self.eyes_closed_since = None
        self.blink_count = 0
        self.alert_count = 0
        self.alarm_active = False
        self._was_closed_seq = False
        self.session_start = time.time()
        self._prev_frame_time = time.time()

        self.attention_tracker = AttentionTracker()
        self.calibrating = True
        self.calib_samples = []
        self.calib_face_samples = []
        self.calib_start_time = time.time()
        self.driver_recognizer = None
        self.driver_recognizer_ready = False
        self._driver_mismatch_count = 0
        self._driver_check_tick = 0
        self._last_face_capture_time = 0.0

        self.start_frame.pack_forget()
        self._build_detection_screen()
        self.det_frame.pack(fill="both", expand=True)

        # Try the real CARLA simulator first; only fall back to the local Tkinter
        # simulation (opened lazily in _update_frame) if no CARLA server responds.
        self.sim_window = None
        self._carla_fallback_started = False
        self.fatigue_strikes = 0
        self.carla_bridge = CarlaBridge()
        self.carla_bridge.start()

        self.running = True
        self._update_frame()

    def _change_driver(self):
        """Resets the cumulative strike count and re-calibrates (including a fresh face
        reference) — brings a vehicle stopped at 3 strikes back into normal operation.
        Triggered either by the manual button or automatically once a different face is
        recognized (see the driver-change check in _update_frame). Does not touch the
        eye-closure/alarm system itself, only the vehicle-response escalation."""
        if not self.running:
            return
        self.fatigue_strikes = 0
        self.eyes_closed_since = None
        self._was_closed_seq = False
        self.alarm_active = False
        self._stop_alarm()
        self.attention_tracker = AttentionTracker()
        self.calibrating = True
        self.calib_samples = []
        self.calib_face_samples = []
        self.calib_start_time = time.time()
        self.driver_recognizer = None
        self.driver_recognizer_ready = False
        self._driver_mismatch_count = 0
        self._last_face_capture_time = 0.0

        if self.carla_bridge is not None:
            self.carla_bridge.announce("DRIVER CHANGED — RESUMING NORMAL SPEED")
        if self.sim_window is not None and not self.sim_window.closed:
            self.sim_window.announce("DRIVER CHANGED — RESUMING NORMAL SPEED")

    def _stop_detection(self):
        self.running = False
        self._stop_alarm()
        if self.session_start is not None and not self.calibrating:
            self._show_session_summary()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.sim_window is not None:
            self.sim_window.close()
            self.sim_window = None
        if self.carla_bridge is not None:
            self.carla_bridge.stop()
            self.carla_bridge = None
        self.det_frame.pack_forget()
        self.det_frame.destroy()
        self.start_frame.pack(fill="both", expand=True)

    def _on_close(self):
        self.running = False
        self._stop_alarm()
        if self.cap is not None:
            self.cap.release()
        if self.sim_window is not None:
            self.sim_window.close()
        if self.carla_bridge is not None:
            self.carla_bridge.stop()
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        self.root.destroy()

    # ── ALARM ─────────────────────────────────────────────
    def _play_alarm(self):
        if not self.alarm_playing:
            pygame.mixer.music.load(ALARM_SOUND)
            pygame.mixer.music.play(-1)
            self.alarm_playing = True

    def _stop_alarm(self):
        if self.alarm_playing:
            pygame.mixer.music.stop()
            self.alarm_playing = False

    # ── INFERENCE ─────────────────────────────────────────
    def _predict_eye(self, eye_img):
        if eye_img is None or eye_img.size == 0:
            return "open", 0.0
        try:
            tensor = EYE_TRANSFORM(eye_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = self.eye_model(tensor)
                probs = torch.softmax(output, dim=1)
                closed_prob = probs[0][0].item()
                state = "closed" if closed_prob > 0.4 else "open"
                return state, closed_prob
        except Exception:
            return "open", 0.0

    def _draw_face_mesh_dots(self, display, lm, w, h):
        """Draws all 468 MediaPipe Face Mesh landmarks as small dots on the video feed.
        These points are computed every frame regardless (they drive the eye-crop and
        head-pose math) but were never actually rendered — this makes that visible.
        The 6 points solvePnP uses for head pose are highlighted larger/brighter so
        it's clear which ones matter for that calculation."""
        for point in lm:
            x, y = int(point.x * w), int(point.y * h)
            cv2.circle(display, (x, y), 1, (0, 255, 210), -1)
        for idx in HEAD_POSE_LANDMARK_IDS:
            x, y = int(lm[idx].x * w), int(lm[idx].y * h)
            cv2.circle(display, (x, y), 3, (0, 165, 255), -1)

    def _detect_face_crop(self, frame):
        """Runs the trained YOLO face detector (first real inference-time use of it in
        this app) to crop the largest detected face, returning a fixed-size grayscale
        image ready for the LBPH recognizer. Only called during calibration and while
        the vehicle is stopped — never during normal per-frame drowsiness detection —
        so it adds no cost to the regular detection loop."""
        try:
            results = self.face_model.predict(frame, verbose=False, conf=0.5)
            if not results or len(results[0].boxes) == 0:
                return None
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            x1, y1, x2, y2 = boxes[int(areas.argmax())].astype(int)
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            return cv2.resize(gray, (200, 200))
        except Exception:
            return None

    # ── MAIN LOOP (Tk-driven, non-blocking) ────────────────
    def _update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(15, self._update_frame)
            return

        display = frame.copy()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mesh_results = self.face_mesh.process(rgb)
        eye_states = []
        left_state = right_state = "—"
        left_conf = right_conf = 0.0
        raw_pose = None

        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                lm = face_landmarks.landmark

                if self.show_face_mesh_var.get():
                    self._draw_face_mesh_dots(display, lm, w, h)

                left_eye_img, (lx1, ly1, lx2, ly2) = get_eye_roi(lm, LEFT_EYE_LANDMARKS, frame)
                right_eye_img, (rx1, ry1, rx2, ry2) = get_eye_roi(lm, RIGHT_EYE_LANDMARKS, frame)

                cv2.rectangle(display, (lx1, ly1), (lx2, ly2), (0, 211, 167), 2)
                cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 211, 167), 2)

                left_state, left_conf = self._predict_eye(left_eye_img)
                right_state, right_conf = self._predict_eye(right_eye_img)
                eye_states.append((left_state, right_state))

                raw_pose = self._get_signed_pose(lm, w, h)

        now = time.time()
        both_closed = any(l == "closed" and r == "closed" for l, r in eye_states)

        if both_closed:
            if self.eyes_closed_since is None:
                self.eyes_closed_since = now
            self._was_closed_seq = True
        else:
            if self._was_closed_seq and self.eyes_closed_since is not None:
                if now - self.eyes_closed_since < EYES_CLOSED_ALERT_SECONDS:
                    self.blink_count += 1
            self.eyes_closed_since = None
            self._was_closed_seq = False
            self.alarm_active = False
            self._stop_alarm()

        closed_duration = (now - self.eyes_closed_since) if self.eyes_closed_since is not None else 0.0

        if closed_duration >= EYES_CLOSED_ALERT_SECONDS:
            if not self.alarm_active:
                self.alarm_active = True
                self.alert_count += 1
                # ── Vehicle-response strike (additive; the alarm above is untouched) ──
                # Each NEW drowsiness episode permanently escalates the vehicle response —
                # it does not reset just because the eyes reopen. Only "Change Driver" does.
                self.fatigue_strikes += 1
            self._play_alarm()
            cv2.rectangle(display, (0, 0), (w, h), (0, 0, 255), 6)

        # FPS
        dt = now - self._prev_frame_time
        self._prev_frame_time = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

        # ── Distraction detection (additive, independent of drowsiness above) ──
        face_visible = bool(mesh_results.multi_face_landmarks)
        if self.calibrating:
            if raw_pose is not None:
                self.calib_samples.append(raw_pose)
            # Also collect a handful of reference face crops (throttled to ~2-3/sec, capped)
            # for the driver-change recognizer, spread across the calibration window.
            if face_visible and len(self.calib_face_samples) < 6 and \
                    now - self._last_face_capture_time >= 0.4:
                crop = self._detect_face_crop(frame)
                if crop is not None:
                    self.calib_face_samples.append(crop)
                    self._last_face_capture_time = now
            if now - self.calib_start_time >= CALIBRATION_SECONDS:
                byaw, bpitch, broll = compute_baseline(self.calib_samples)
                self.attention_tracker.calibrate(byaw, bpitch, broll)
                self.calibrating = False
                if len(self.calib_face_samples) >= MIN_CALIB_FACE_SAMPLES:
                    self.driver_recognizer = cv2.face.LBPHFaceRecognizer_create()
                    self.driver_recognizer.train(
                        self.calib_face_samples, np.array([0] * len(self.calib_face_samples)))
                    self.driver_recognizer_ready = True
                else:
                    self.driver_recognizer_ready = False
            attention_status = {
                "state": "CALIBRATING", "direction": "—",
                "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                "duration": 0.0, "score": 100.0, "events": 0,
            }
        else:
            attention_status = self.attention_tracker.update(raw_pose, face_visible, now)

        # ── Automatic driver-change detection (additive; only runs while stopped) ──
        if (self.fatigue_strikes >= FATIGUE_STOP_STRIKES and self.driver_recognizer_ready
                and not self.calibrating):
            self._driver_check_tick += 1
            if self._driver_check_tick % DRIVER_CHECK_EVERY_N_FRAMES == 0:
                crop = self._detect_face_crop(frame)
                if crop is not None:
                    _, confidence = self.driver_recognizer.predict(crop)
                    if confidence > DRIVER_CHANGE_LBPH_THRESHOLD:
                        self._driver_mismatch_count += 1
                    else:
                        self._driver_mismatch_count = 0
                    if self._driver_mismatch_count >= DRIVER_CHANGE_CONFIRM_COUNT:
                        self._change_driver()
        else:
            self._driver_mismatch_count = 0

        self._draw_attention_overlay(display, attention_status)
        self._render_frame(display)
        self._update_sidebar(left_state, left_conf, right_state, right_conf, closed_duration)
        self._update_attention_sidebar(attention_status)

        # ── Vehicle response: real CARLA if reachable, local fallback otherwise.
        # All status/speed messaging is drawn inside CARLA (or the fallback sim window),
        # never on this dashboard. ──
        if self.carla_bridge is not None:
            if self.carla_bridge.connected:
                self.carla_bridge.set_driver_state(
                    self.fatigue_strikes, attention_status["state"], attention_status["direction"])
            elif self.carla_bridge.error and not self._carla_fallback_started:
                self._carla_fallback_started = True
                self.sim_window = VehicleSimulation(self.root)
                self.root.update_idletasks()
                main_x = self.root.winfo_x()
                main_w = self.root.winfo_width()
                self.sim_window.top.geometry(f"+{main_x + main_w + 10}+{self.root.winfo_y()}")

        if self.sim_window is not None and not self.sim_window.closed and not self.calibrating:
            self.sim_window.update(self.fatigue_strikes, attention_status["state"], attention_status["direction"])

        self.root.after(15, self._update_frame)

    def _render_frame(self, display_bgr):
        rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        target_w = max(self.video_label.winfo_width(), 480)
        target_h = max(self.video_label.winfo_height(), 360)
        img_ratio = img.width / img.height
        box_ratio = target_w / target_h
        if img_ratio > box_ratio:
            new_w = target_w
            new_h = int(target_w / img_ratio)
        else:
            new_h = target_h
            new_w = int(target_h * img_ratio)
        img = img.resize((max(new_w, 1), max(new_h, 1)))

        photo = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=photo)
        self.video_label.image = photo

    def _update_sidebar(self, left_state, left_conf, right_state, right_conf, closed_duration=0.0):
        if self.alarm_active:
            self.status_badge.config(text="DROWSY ALERT!", bg=ALERT, fg="#ffffff")
        else:
            self.status_badge.config(text="AWAKE", bg=ACCENT, fg="#0f1117")

        self.eye_l_val.config(
            text=f"{left_state.upper()} ({left_conf:.2f})",
            fg=ALERT if left_state == "closed" else ACCENT)
        self.eye_r_val.config(
            text=f"{right_state.upper()} ({right_conf:.2f})",
            fg=ALERT if right_state == "closed" else ACCENT)

        self.meter["value"] = min(closed_duration, EYES_CLOSED_ALERT_SECONDS)

        elapsed = int(time.time() - self.session_start)
        mm, ss = divmod(elapsed, 60)
        self.session_val.config(text=f"{mm:02d}:{ss:02d}")
        self.blink_val.config(text=str(self.blink_count))
        self.alert_val.config(text=str(self.alert_count), fg=ALERT if self.alert_count else TEXT)
        self.fps_val.config(text=f"{self.fps:.1f}")


def main():
    root = tk.Tk()
    FatigueApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
