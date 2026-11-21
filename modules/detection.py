
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "yolo26s.pt"


def load_model():
    """Load YOLO26 Small, using local weights if available."""
    if MODEL_PATH.exists():
        return YOLO(str(MODEL_PATH))

    return YOLO("yolo26s.pt")


def detect_image(model, image, confidence=0.25, image_size=640):
    """
    Run object detection on an RGB image.

    Returns:
        annotated_image: RGB image with bounding boxes
        detections: list of detected objects
    """

    results = model.predict(
        source=image,
        conf=confidence,
        imgsz=image_size,
        verbose=False
    )

    result = results[0]

    # Ultralytics returns annotated images in BGR format
    annotated_bgr = result.plot()

    # Convert BGR to RGB for Streamlit
    annotated_rgb = cv2.cvtColor(
        annotated_bgr,
        cv2.COLOR_BGR2RGB
    )

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            score = float(box.conf[0].item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "Object": class_name,
                "Confidence": round(score, 4),
                "X1": round(x1, 2),
                "Y1": round(y1, 2),
                "X2": round(x2, 2),
                "Y2": round(y2, 2)
            })

    return annotated_rgb, detections
