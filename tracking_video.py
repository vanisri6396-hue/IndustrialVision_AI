
from pathlib import Path
import cv2
import time
from ultralytics import YOLO

# ==================================================
# 1. DEFINE INPUT AND OUTPUT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

VIDEO_PATH = PROJECT_DIR / "walking_video.mp4"
OUTPUT_PATH = PROJECT_DIR / "outputs" / "tracked_factory_video.mp4"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Input video:", VIDEO_PATH)
print("Output video:", OUTPUT_PATH)


# ==================================================
# 2. CHECK INPUT VIDEO
# ==================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")


# ==================================================
# 3. LOAD YOLO26 MODEL
# ==================================================

model = YOLO("yolo26s.pt")

print("\nYOLO26 Small model loaded successfully!")


# ==================================================
# 4. OPEN INPUT VIDEO
# ==================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open input video!")

video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if video_fps <= 0:
    video_fps = 30.0

print("\n========== VIDEO INFORMATION ==========")
print("Width:", video_width)
print("Height:", video_height)
print("FPS:", round(video_fps, 2))
print("Total frames:", total_frames)


# ==================================================
# 5. CREATE OUTPUT VIDEO WRITER
# ==================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    video_fps,
    (video_width, video_height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video!")


# ==================================================
# 6. CREATE DISPLAY WINDOW
# ==================================================

WINDOW_NAME = "IndustrialVision AI - Object Tracking"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)


# ==================================================
# 7. INITIALIZE TRACKING VARIABLES
# ==================================================

frame_count = 0
total_detections = 0

# Store unique tracking IDs
unique_track_ids = set()

# Store detected object classes
class_counts = {}

start_time = time.time()

print("\nStarting YOLO26 + ByteTrack tracking...")
print("Press Q to stop.")


# ==================================================
# 8. PROCESS VIDEO FRAME BY FRAME
# ==================================================

try:

    while True:

        # Read video frame
        ret, frame = cap.read()

        if not ret:
            print("\nVideo processing completed.")
            break

        frame_count += 1

        # Run YOLO object tracking
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        # Draw bounding boxes and tracking IDs
        annotated_frame = result.plot()

        # ==================================================
        # 9. PROCESS TRACKING RESULTS
        # ==================================================

        if result.boxes is not None and len(result.boxes) > 0:

            total_detections += len(result.boxes)

            # Tracking IDs are available when assigned
            if result.boxes.id is not None:

                track_ids = result.boxes.id.int().cpu().tolist()

                class_ids = result.boxes.cls.int().cpu().tolist()

                confidences = (
                    result.boxes.conf.cpu().tolist()
                )

                for track_id, class_id, confidence in zip(
                    track_ids, class_ids, confidences
                ):

                    class_name = model.names[class_id]

                    # Store unique tracking IDs
                    unique_track_ids.add(track_id)

                    # Count detections by class
                    class_counts[class_name] = (
                        class_counts.get(class_name, 0) + 1
                    )

                    # Print tracking information
                    print(
                        f"Frame: {frame_count} | "
                        f"ID: {track_id} | "
                        f"Object: {class_name} | "
                        f"Confidence: {confidence:.3f}"
                    )

        # ==================================================
        # 10. CALCULATE PROCESSING FPS
        # ==================================================

        elapsed_time = time.time() - start_time

        processing_fps = (
            frame_count / elapsed_time
            if elapsed_time > 0 else 0
        )

        # Display information on video
        cv2.putText(
            annotated_frame,
            f"FPS: {processing_fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Unique IDs: {len(unique_track_ids)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # ==================================================
        # 11. SAVE PROCESSED FRAME
        # ==================================================

        writer.write(annotated_frame)

        # ==================================================
        # 12. RESIZE FRAME FOR DISPLAY
        # ==================================================

        frame_height, frame_width = annotated_frame.shape[:2]

        max_width = 1280
        max_height = 720

        scale = min(
            max_width / frame_width,
            max_height / frame_height,
            1.0
        )

        if scale < 1.0:

            display_frame = cv2.resize(
                annotated_frame,
                (
                    int(frame_width * scale),
                    int(frame_height * scale)
                ),
                interpolation=cv2.INTER_AREA
            )

        else:
            display_frame = annotated_frame

        cv2.imshow(WINDOW_NAME, display_frame)

        # ==================================================
        # 13. DISPLAY PROGRESS
        # ==================================================

        if frame_count % 100 == 0:

            print(
                f"Processed {frame_count}/{total_frames} frames"
            )

        # Quit using Q
        if cv2.waitKey(1) & 0xFF == ord("q"):

            print("\nTracking stopped by user.")
            break


finally:

    # ==================================================
    # 14. RELEASE RESOURCES
    # ==================================================

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


# ==================================================
# 15. DISPLAY FINAL TRACKING SUMMARY
# ==================================================

total_time = time.time() - start_time

average_fps = (
    frame_count / total_time
    if total_time > 0 else 0
)

print("\n========== TRACKING SUMMARY ==========")

print("Frames processed:", frame_count)
print("Total detections:", total_detections)
print("Unique tracking IDs:", len(unique_track_ids))
print("Processing time:", round(total_time, 2), "seconds")
print("Average processing FPS:", round(average_fps, 2))

print("\nDetection counts by class:")

for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

print("\nTracked video saved at:")
print(OUTPUT_PATH)

if OUTPUT_PATH.exists():
    print("File size:", OUTPUT_PATH.stat().st_size, "bytes")

print("\nOBJECT TRACKING COMPLETED!")
