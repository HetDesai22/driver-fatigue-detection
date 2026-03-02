import cv2
import torch
import numpy as np
import pygame
import mediapipe as mp
from ultralytics import YOLO
import torch.nn as nn
from torchvision import transforms

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
FACE_MODEL_PATH = r"E:\driver-fatigue-detection\models\face_detector\train3\weights\best.pt"
EYE_MODEL_PATH  = r"E:\driver-fatigue-detection\models\eye_classifier\eye_model.pth"
ALARM_SOUND     = r"E:\driver-fatigue-detection\alarm.wav"
CONSEC_FRAMES   = 15
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────
# CNN EYE MODEL
# ─────────────────────────────
class EyeCNN(nn.Module):
    def __init__(self):
        super(EyeCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

# ─────────────────────────────
# LOAD MODELS
# ─────────────────────────────
print("Loading models...")
face_model = YOLO(FACE_MODEL_PATH)
print("✅ Face detector loaded")

eye_model = EyeCNN().to(DEVICE)
eye_model.load_state_dict(torch.load(EYE_MODEL_PATH, map_location=DEVICE))
eye_model.eval()
print("✅ Eye classifier loaded")

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("✅ MediaPipe loaded")

# ─────────────────────────────
# ALARM
# ─────────────────────────────
pygame.mixer.init()
alarm_playing = False

def play_alarm():
    global alarm_playing
    if not alarm_playing:
        pygame.mixer.music.load(ALARM_SOUND)
        pygame.mixer.music.play(-1)
        alarm_playing = True

def stop_alarm():
    global alarm_playing
    if alarm_playing:
        pygame.mixer.music.stop()
        alarm_playing = False

# ─────────────────────────────
# EYE TRANSFORM
# ─────────────────────────────
eye_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((24, 24)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_eye(eye_img):
    if eye_img is None or eye_img.size == 0:
        return "open", 0.0
    try:
        tensor = eye_transform(eye_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output  = eye_model(tensor)
            probs   = torch.softmax(output, dim=1)
            closed_prob = probs[0][0].item()  # probability of closed
            # Lower threshold = more sensitive to partial closing
            state = "closed" if closed_prob > 0.4 else "open"
            return state, closed_prob
    except:
        return "open", 0.0

# MediaPipe eye landmark indices
# Left eye:  33, 160, 158, 133, 153, 144
# Right eye: 362, 385, 387, 263, 373, 380
LEFT_EYE_LANDMARKS  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

def get_eye_roi(landmarks, indices, frame, padding=10):
    h, w = frame.shape[:2]
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x1 = max(0, min(p[0] for p in points) - padding)
    y1 = max(0, min(p[1] for p in points) - padding)
    x2 = min(w, max(p[0] for p in points) + padding)
    y2 = min(h, max(p[1] for p in points) + padding)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

# ─────────────────────────────
# MAIN LOOP
# ─────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

closed_frame_count = 0
print("\n✅ Detection started! Press Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display  = frame.copy()
    h, w     = frame.shape[:2]
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── MediaPipe Face Mesh ──
    mesh_results = face_mesh.process(rgb)
    eye_states   = []

    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            lm = face_landmarks.landmark

            # ── Extract Eye ROIs ──
            left_eye_img,  (lx1, ly1, lx2, ly2) = get_eye_roi(lm, LEFT_EYE_LANDMARKS,  frame)
            right_eye_img, (rx1, ry1, rx2, ry2) = get_eye_roi(lm, RIGHT_EYE_LANDMARKS, frame)

            # Draw eye boxes
            cv2.rectangle(display, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)

            # ── Predict Eye State ──
            left_state,  left_conf  = predict_eye(left_eye_img)
            right_state, right_conf = predict_eye(right_eye_img)
            eye_states.append((left_state, right_state))

            # Labels
            cl = (0, 0, 255) if left_state  == "closed" else (0, 255, 0)
            cr = (0, 0, 255) if right_state == "closed" else (0, 255, 0)
            cv2.putText(display, f"L:{left_state}({left_conf:.2f})",(lx1, ly1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cl, 1)
            cv2.putText(display, f"R:{right_state}({right_conf:.2f})",(rx1, ry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cr, 1)

    # ── Drowsiness Logic ──
    both_closed = any(l == "closed" and r == "closed" for l, r in eye_states)

    if both_closed:
        closed_frame_count += 1
    else:
        closed_frame_count = 0
        stop_alarm()

    if closed_frame_count >= CONSEC_FRAMES:
        play_alarm()
        cv2.rectangle(display, (0, 0), (w, h), (0, 0, 255), 4)
        cv2.putText(display, "DROWSINESS DETECTED!",
                    (w//2 - 180, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # ── Status ──
    status = f"Closed frames: {closed_frame_count}/{CONSEC_FRAMES}"
    color  = (0, 0, 255) if closed_frame_count >= CONSEC_FRAMES else (0, 255, 0)
    cv2.putText(display, status, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Driver Fatigue Detection", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stop_alarm()
pygame.mixer.quit()