"""
Driver Fatigue Detection System — Desktop Dashboard
Pipeline: YOLO face detector -> MediaPipe face mesh (eye landmarks)
          -> CNN eye-state classifier (open/closed) -> drowsiness alarm
"""

import time
import threading
import tkinter as tk
from tkinter import ttk

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
CONSEC_FRAMES   = 15
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEFT_EYE_LANDMARKS  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

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


class FatigueApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Fatigue Detection System")
        self.root.geometry("1200x700")
        self.root.minsize(1040, 640)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # runtime state
        self.cap = None
        self.running = False
        self.face_model = None
        self.eye_model = None
        self.face_mesh = None
        self.models_ready = False
        self.alarm_playing = False

        self.closed_frame_count = 0
        self.blink_count = 0
        self.alert_count = 0
        self.alarm_active = False
        self._was_closed_seq = False
        self.session_start = None
        self._prev_frame_time = time.time()
        self.fps = 0.0

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

        sidebar = tk.Frame(self.det_frame, bg=PANEL_BG, width=300)
        sidebar.pack(side="right", fill="y", padx=(8, 16), pady=16)
        sidebar.pack_propagate(False)

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
                                      maximum=CONSEC_FRAMES, length=260)
        self.meter.pack(padx=20, pady=(0, 20))

        self._sep(sidebar)

        self.session_val   = self._metric_row(sidebar, "Session Time")
        self.blink_val     = self._metric_row(sidebar, "Blinks")
        self.alert_val     = self._metric_row(sidebar, "Alerts Triggered")
        self.fps_val       = self._metric_row(sidebar, "FPS")

        btn_wrap = tk.Frame(sidebar, bg=PANEL_BG)
        btn_wrap.pack(side="bottom", fill="x", padx=20, pady=20)
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

        self.closed_frame_count = 0
        self.blink_count = 0
        self.alert_count = 0
        self.alarm_active = False
        self._was_closed_seq = False
        self.session_start = time.time()
        self._prev_frame_time = time.time()

        self.start_frame.pack_forget()
        self._build_detection_screen()
        self.det_frame.pack(fill="both", expand=True)

        self.running = True
        self._update_frame()

    def _stop_detection(self):
        self.running = False
        self._stop_alarm()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.det_frame.pack_forget()
        self.det_frame.destroy()
        self.start_frame.pack(fill="both", expand=True)

    def _on_close(self):
        self.running = False
        self._stop_alarm()
        if self.cap is not None:
            self.cap.release()
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

        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                lm = face_landmarks.landmark
                left_eye_img, (lx1, ly1, lx2, ly2) = get_eye_roi(lm, LEFT_EYE_LANDMARKS, frame)
                right_eye_img, (rx1, ry1, rx2, ry2) = get_eye_roi(lm, RIGHT_EYE_LANDMARKS, frame)

                cv2.rectangle(display, (lx1, ly1), (lx2, ly2), (0, 211, 167), 2)
                cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 211, 167), 2)

                left_state, left_conf = self._predict_eye(left_eye_img)
                right_state, right_conf = self._predict_eye(right_eye_img)
                eye_states.append((left_state, right_state))

        both_closed = any(l == "closed" and r == "closed" for l, r in eye_states)

        if both_closed:
            self.closed_frame_count += 1
            self._was_closed_seq = True
        else:
            if self._was_closed_seq and self.closed_frame_count < CONSEC_FRAMES:
                self.blink_count += 1
            self.closed_frame_count = 0
            self._was_closed_seq = False
            self.alarm_active = False
            self._stop_alarm()

        if self.closed_frame_count >= CONSEC_FRAMES:
            if not self.alarm_active:
                self.alarm_active = True
                self.alert_count += 1
            self._play_alarm()
            cv2.rectangle(display, (0, 0), (w, h), (0, 0, 255), 6)

        # FPS
        now = time.time()
        dt = now - self._prev_frame_time
        self._prev_frame_time = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

        self._render_frame(display)
        self._update_sidebar(left_state, left_conf, right_state, right_conf)

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

    def _update_sidebar(self, left_state, left_conf, right_state, right_conf):
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

        self.meter["value"] = min(self.closed_frame_count, CONSEC_FRAMES)

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
