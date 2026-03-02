from ultralytics import YOLO
import os

DATASET_YAML = r"E:\driver-fatigue-detection\dataset\faces\dataset.yaml"
MODEL_SAVE   = r"E:\driver-fatigue-detection\models\face_detector"
EPOCHS       = 50
IMG_SIZE     = 640
BATCH_SIZE   = 16

yaml_content = """
path: E:/driver-fatigue-detection/dataset/faces
train: images/train
val: images/val

nc: 1
names: ['face']
"""

os.makedirs(os.path.dirname(DATASET_YAML), exist_ok=True)
with open(DATASET_YAML, 'w') as f:
    f.write(yaml_content)
print("✅ dataset.yaml created")

# ─────────────────────────────
# THIS IS THE FIX ↓
# ─────────────────────────────
if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    print("✅ YOLOv8 model loaded")

    results = model.train(
        data    = DATASET_YAML,
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH_SIZE,
        device  = 0,
        project = MODEL_SAVE,
        name    = 'train',
        patience= 10,
        save    = True,
        plots   = True,
        workers = 0          # ← also add this for Windows
    )

    print("\nTraining Complete! ✅")
    print(f"Best model: {MODEL_SAVE}/train/weights/best.pt")