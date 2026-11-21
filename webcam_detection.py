
import cv2
import time
from pathlib import Path
from ultralytics import YOLO

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Load YOLO26 Small model
model = YOLO("yolo26s.pt")

print("YOLO26 model loaded!")
print("Starting webcam...")

# Open default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not access webcam!")

WINDOW_NAME = "IndustrialVision AI - Live Detection"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

frame_count = 0
total_detections = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read webcam frame.")
            break

        frame_count += 1

        # Run YOLO detection
        results = model.predict(
            source=frame,
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        # Draw bounding boxes and labels
        annotated_frame = result.plot()

        # Count detected objects
        detection_count = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        total_detections += detection_count

        # Calculate processing FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Display information
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Objects: {detection_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display webcam output
        cv2.imshow(WINDOW_NAME, annotated_frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

print("\n========== WEBCAM SUMMARY ==========")
print("Frames processed:", frame_count)
print("Total detections:", total_detections)
print("Processing FPS:", round(fps, 2) if frame_count else 0)
print("Webcam detection completed!")
