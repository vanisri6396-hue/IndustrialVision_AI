
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "datasets" / "construction-ppe"

ORIGINAL_YAML = DATASET_DIR / "data.yaml"
FIXED_YAML = DATASET_DIR / "data_ppe_fixed.yaml"

MODEL_NAME = "yolo26s.pt"

EPOCHS = 80
IMAGE_SIZE = 640
BATCH_SIZE = 8
PATIENCE = 20
WORKERS = 4

RUN_NAME = "construction_ppe_yolo26s"

RUN_DIR = (
    PROJECT_DIR / "runs" / "finetuning" / RUN_NAME
)

BEST_MODEL_PATH = RUN_DIR / "weights" / "best.pt"

# ==========================================
# 2. VERIFY DATASET
# ==========================================

print("\n========== DATASET VERIFICATION ==========")

if not ORIGINAL_YAML.exists():
    raise FileNotFoundError(ORIGINAL_YAML)

for split in ["train", "val"]:
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            f"Missing image or label directory: {split}"
        )

    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ]

    labels = list(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing = image_stems - label_stems
    extra = label_stems - image_stems

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Labels:", len(labels))

    if missing or extra:
        raise RuntimeError(
            f"Image-label mismatch in {split}. "
            f"Missing labels: {len(missing)}, "
            f"Extra labels: {len(extra)}"
        )

# ==========================================
# 3. PREPARE YAML
# ==========================================

print("\n========== PREPARING DATASET YAML ==========")

with open(ORIGINAL_YAML, "r", encoding="utf-8") as f:
    data_config = yaml.safe_load(f)

names = data_config["names"]

if isinstance(names, dict):
    names = [names[i] for i in range(len(names))]

data_config["names"] = names
data_config["nc"] = len(names)
data_config["path"] = str(DATASET_DIR.resolve())
data_config["train"] = "images/train"
data_config["val"] = "images/val"

print("Classes:", names)
print("Number of classes:", len(names))

with open(FIXED_YAML, "w", encoding="utf-8") as f:
    yaml.safe_dump(data_config, f, sort_keys=False)

# ==========================================
# 4. SELECT DEVICE
# ==========================================

device = 0 if torch.cuda.is_available() else "cpu"

print("\n========== TRAINING DEVICE ==========")

if device == 0:
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA unavailable. Using CPU.")

# ==========================================
# 5. LOAD PRETRAINED YOLO MODEL
# ==========================================

print("\n========== LOADING YOLO26S ==========")

model = YOLO(MODEL_NAME)

print("Pretrained model loaded:", MODEL_NAME)

# ==========================================
# 6. FINE-TUNING
# ==========================================

print("\n========== STARTING TRAINING ==========")

results = model.train(
    data=str(FIXED_YAML),

    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,

    patience=PATIENCE,
    workers=WORKERS,

    device=device,

    project=str(PROJECT_DIR / "runs" / "finetuning"),
    name=RUN_NAME,
    exist_ok=True,

    pretrained=True,
    optimizer="auto",

    cos_lr=True,
    close_mosaic=10,

    save=True,
    save_period=10,
    plots=True,
    verbose=True
)

# ==========================================
# 7. CHECK BEST MODEL
# ==========================================

print("\n========== TRAINING COMPLETED ==========")

if not BEST_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Best weights not found: {BEST_MODEL_PATH}"
    )

print("Best model:", BEST_MODEL_PATH)

best_model = YOLO(str(BEST_MODEL_PATH))

# ==========================================
# 8. MODEL VALIDATION
# ==========================================

print("\n========== VALIDATING PPE MODEL ==========")

metrics = best_model.val(
    data=str(FIXED_YAML),
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=device,
    plots=True
)

# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("\n========== FINAL PPE MODEL RESULTS ==========")

print(f"Precision:  {metrics.box.mp:.4f}")
print(f"Recall:     {metrics.box.mr:.4f}")
print(f"mAP50:      {metrics.box.map50:.4f}")
print(f"mAP50-95:   {metrics.box.map:.4f}")

print("\n========== SUCCESS ==========")
print("Fine-tuning completed!")
print("Model location:", BEST_MODEL_PATH)
