from ultralytics import YOLO

def main():

    print("Loading YOLO26 Small model...")

    model = YOLO("yolo26s.pt")

    print("Model loaded successfully!")

    print("\nStarting model evaluation...")
    print("Dataset: COCO8")
    
    # Validate the pretrained model
    metrics = model.val(
        data="coco8.yaml",
        imgsz=640,
        plots=True,
        workers=0,
        verbose=True
    )

    # Display evaluation metrics
    print("\n========== MODEL EVALUATION RESULTS ==========")

    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    print("\n========== PROCESSING SPEED ==========")

    print("Preprocessing:", metrics.speed["preprocess"], "ms/image")
    print("Inference:", metrics.speed["inference"], "ms/image")
    print("Postprocessing:", metrics.speed["postprocess"], "ms/image")

    print("\n========== EVALUATION COMPLETED ==========")
    print("Check the runs/detect/ folder for evaluation plots.")

if __name__ == "__main__":
    main()
