
from pathlib import Path
import cv2
from ultralytics import YOLO

# ==================================================
# 1. DEFINE INPUT AND OUTPUT PATHS
# ==================================================

IMAGE_PATH = "factory.jpg"
OUTPUT_PATH = "outputs/detected_factory.jpg"


# ==================================================
# 2. CREATE OUTPUT FOLDER
# ==================================================

Path("outputs").mkdir(parents=True, exist_ok=True)


# ==================================================
# 3. LOAD PRETRAINED YOLO26 MODEL
# ==================================================

model = YOLO("yolo26s.pt")

print("\nYOLO26 Small model loaded successfully!")


# ==================================================
# 4. RUN OBJECT DETECTION
# ==================================================

results = model.predict(
    source=IMAGE_PATH,
    conf=0.60,
    imgsz=960,
    verbose=False,
    iou=0.5,
    nms=True,
)


# ==================================================
# 5. PROCESS DETECTION RESULTS
# ==================================================

for result in results:

    # Draw bounding boxes, class labels and confidence scores
    annotated_image = result.plot()

    print("\n========== DETECTION RESULTS ==========")

    # Check whether objects were detected
    if result.boxes is None or len(result.boxes) == 0:

        print("No objects detected.")

    else:

        for i, box in enumerate(result.boxes):

            # Get class ID
            class_id = int(box.cls[0].item())

            # Get object name
            class_name = model.names[class_id]

            # Get confidence score
            confidence = float(box.conf[0].item())

            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Calculate bounding box dimensions
            width = x2 - x1
            height = y2 - y1

            # Display detection details
            print(f"\nObject {i + 1}: {class_name}")
            print(f"Confidence: {confidence:.4f}")

            print(f"X1 (Left):   {x1:.2f}")
            print(f"Y1 (Top):    {y1:.2f}")
            print(f"X2 (Right):  {x2:.2f}")
            print(f"Y2 (Bottom): {y2:.2f}")

            print(f"Width:       {width:.2f}")
            print(f"Height:      {height:.2f}")

            print("--------------------------------------")

    # ==================================================
    # 6. SAVE ANNOTATED IMAGE
    # ==================================================

    success = cv2.imwrite(OUTPUT_PATH, annotated_image)

    if success:
        print("\nAnnotated image saved successfully!")
        print("Output:", OUTPUT_PATH)
    else:
        print("\nError: Could not save the output image.")


# ==================================================
# 7. COMPLETION MESSAGE
# ==================================================

print("\n======================================")
print("OBJECT DETECTION COMPLETED!")
print("======================================")
