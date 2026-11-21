# 🏭 IndustrialVision AI
### Real-Time Object Detection in Industrial Applications using Ultralytics YOLO

## 📌 Project Overview

**IndustrialVision AI** is a computer vision project developed as part of an Ultralytics YOLO technical seminar titled **"Real-Time Object Detection in Industrial Applications."**

The project explores how deep learning and real-time object detection can be applied to industrial environments such as factories, manufacturing facilities, warehouses and construction sites.

Using the Ultralytics YOLO framework, the system detects objects in images and videos, identifies their classes, displays bounding boxes and confidence scores, and performs object tracking across video frames.

The project initially uses pretrained YOLO weights for general object detection and is being extended through custom dataset training to recognize construction Personal Protective Equipment (PPE).

The long-term objective is to develop an intelligent industrial monitoring application that supports workplace safety and visual inspection.

## 🎯 Project Objectives

- Implement real-time object detection using Ultralytics YOLO.
- Detect and classify objects in industrial environments.
- Perform object detection on images and video streams.
- Display bounding boxes, class labels and confidence scores.
- Track detected objects across video frames.
- Understand pretrained deep learning models and transfer learning.
- Fine-tune YOLO using a custom construction PPE dataset.
- Evaluate model performance using standard object detection metrics.
- Develop an application for industrial safety monitoring.

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Ultralytics YOLO26 | Object detection and tracking |
| PyTorch | Deep learning framework |
| OpenCV | Image and video processing |
| NumPy | Numerical operations |
| Google Colab | GPU-based model training |
| Google Drive | Dataset and model storage |
| VS Code | Development environment |
| Git & GitHub | Version control |

## ⚙️ System Architecture

The project follows a computer vision pipeline:

```text
            Input
              |
       +------+------+
       |             |
     Images        Videos
       |             |
       +------+------+
              |
       Image/Video Processing
              |
        Pretrained YOLO
              |
       Object Detection
              |
    +---------+----------+
    |         |          |
 Bounding   Class     Confidence
  Boxes     Labels      Scores
    |         |          |
    +---------+----------+
              |
        Object Tracking
          (Videos)
              |
       Annotated Output
              |
       Result Visualization
```

**Custom model training extension:**

```text
Construction PPE Dataset
          |
    Dataset Validation
          |
     YOLO Training
          |
    Fine-tuned Weights
          |
    Model Evaluation
          |
  Industrial PPE Detection
```

## 🔍 Implemented Features

### 1. Image Object Detection

- Loads an input image.
- Runs YOLO object detection.
- Identifies recognized objects.
- Draws bounding boxes around detected objects.
- Displays class names and confidence scores.
- Saves the annotated image.

### 2. Video Object Detection

- Processes video frame by frame.
- Detects objects in each frame.
- Displays detection results.
- Generates an annotated output video.

### 3. Object Tracking

- Tracks detected objects across consecutive frames.
- Assigns tracking IDs to detected objects.
- Helps maintain object identities throughout a video.
- Saves the tracking output.

### 4. Model Evaluation

The initial pretrained model was evaluated using the COCO8 dataset.

Evaluation metrics include:

- Precision
- Recall
- mAP50
- mAP50-95
- Preprocessing time
- Inference time
- Postprocessing time

### 5. Custom PPE Dataset Preparation

A Construction PPE dataset was selected to extend the project toward industrial workplace safety monitoring.

Dataset preparation includes:

- Dataset extraction.
- Training and validation separation.
- YOLO annotation format verification.
- Image and label correspondence checks.
- Class distribution inspection.

### 6. YOLO Fine-tuning (In Progress)

The project is being extended through transfer learning using the pretrained `yolo26s.pt` model and the custom Construction PPE dataset.

The training workflow uses Google Colab with an NVIDIA Tesla T4 GPU.

The objective is to adapt the pretrained model to recognize PPE-related classes from the custom dataset.

**Note:** Dataset validation has been completed. Actual custom model training and fine-tuned model evaluation are still pending.

## 📊 Initial Model Evaluation

The pretrained YOLO model was evaluated using COCO8.

| Evaluation Metric | Result |
|---|---:|
| Precision | 0.7940 |
| Recall | 0.8667 |
| mAP50 | 0.9170 |
| mAP50-95 | 0.6933 |
| Preprocessing | 1.1402 ms/image |
| Inference | 116.8452 ms/image |
| Postprocessing | 1.8165 ms/image |

**Important:** These are the initial evaluation results on COCO8. They are not results from the custom Construction PPE dataset or the fine-tuned model.

## 📁 Project Structure

```text
IndustrialVision_AI/
│
├── datasets/
│   └── construction-ppe/
│       ├── data.yaml
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
│
├── outputs/
│   ├── detected_factory.jpg
│   └── detected_factory_video.mp4
│
├── image_detection.py
├── video_detection.py
├── object_tracking.py
├── validate_dataset.py
│
├── factory.jpg
├── factory_video.mp4
│
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: Training scripts, model weights and additional application files will be added as development progresses. Dataset and output folders may be excluded from GitHub when appropriate.*

## 💻 Installation and Setup

### Prerequisites

- Python 3.10 or later
- Visual Studio Code
- Git
- Internet connection

### 1. Clone the Repository

```bash
git clone https://github.com/vanisri6396-hue/IndustrialVision_AI.git
```

```bash
cd IndustrialVision_AI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install ultralytics opencv-python numpy
```

### 4. Verify YOLO Installation

```bash
yolo version
```

## 🚀 Running the Project

### Image Detection

```bash
python image_detection.py
```

The script loads an image, performs detection and saves the annotated output.

### Video Detection

```bash
python video_detection.py
```

The script processes the input video and generates an annotated video.

### Object Tracking

```bash
python object_tracking.py
```

The tracking implementation detects and tracks objects across video frames using YOLO tracking functionality.

### Dataset Validation

```bash
python validate_dataset.py
```

The validation script checks:

- Number of training images.
- Number of validation images.
- Number of annotation files.
- Missing annotation files.
- Extra annotation files.
- Class distribution.

## 📦 Custom Dataset Details

**Dataset:** Construction PPE Detection Dataset

**Purpose:** Train a custom object detection model for construction and industrial safety monitoring.

| Dataset Component | Count |
|---|---:|
| Training images | 800 |
| Validation images | 200 |
| Total images | 1,000 |
| Training label files | 800 |
| Validation label files | 200 |
| Missing label files | 0 |
| Extra label files | 0 |

The dataset uses YOLO-format annotations.

### Dataset Configuration

The `data.yaml` file specifies:

- Training image directory.
- Validation image directory.
- Number of object classes.
- Class names.

The dataset was validated using a custom Python script before starting the training stage.

## 🏋️ Model Fine-tuning Workflow

The custom training process follows these stages:

1. Load pretrained YOLO26s weights.
2. Load the custom Construction PPE dataset.
3. Configure training parameters.
4. Train the model using Google Colab GPU.
5. Monitor training and validation losses.
6. Save the best-performing model.
7. Evaluate the trained model.
8. Test on unseen images and videos.

### Training Environment

| Component | Configuration |
|---|---|
| Platform | Google Colab |
| GPU | NVIDIA Tesla T4 |
| Framework | PyTorch |
| CUDA | Available |
| Initial weights | yolo26s.pt |
| Dataset | Construction PPE |

**Training status:** Dataset prepared and validated; custom training not yet completed.

## 📈 Evaluation Metrics Explained

| Metric | Description |
|---|---|
| Precision | Measures how many predicted detections are correct. |
| Recall | Measures how many actual objects are successfully detected. |
| mAP50 | Mean Average Precision at IoU 0.50. |
| mAP50-95 | Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95. |
| Training Loss | Measures model prediction errors during training. |
| Validation Loss | Measures errors on validation data. |

The custom model will be evaluated using these metrics after training.

## 🌍 Real-World Applications

Potential applications of IndustrialVision AI include:

- Construction site safety monitoring.
- Helmet and safety vest detection.
- Industrial equipment monitoring.
- Factory object detection.
- Warehouse monitoring.
- Worker safety compliance.
- Visual inspection and automated surveillance.

These are intended application areas; the current implementation does not yet provide a complete production-ready industrial safety monitoring system.

## 🔮 Future Enhancements

The following enhancements are planned to make IndustrialVision AI more practical and impactful:

### 1. Custom YOLO Model
Fine-tune YOLO using industrial and construction safety datasets.

### 2. PPE Compliance Monitoring
Detect missing safety helmets and safety vests.

### 3. Real-Time Camera Integration
Connect CCTV cameras or live video streams for continuous monitoring.

### 4. Automated Safety Alerts
Generate alerts when predefined PPE violations are detected.

### 5. Industrial Monitoring Dashboard
Build a web application to display detection results, monitoring information and safety events.

### 6. Event Logging and Analytics
Store safety incidents, timestamps and detection information for analysis.

### 7. Database Integration
Use a database to maintain detection history and safety records.

### 8. Application Deployment
Deploy the application for practical industrial monitoring scenarios.

## ⚠️ Limitations

- Detection accuracy depends on image quality, lighting and camera angle.
- General pretrained models may not correctly recognize specialized industrial objects.
- Custom model performance depends on dataset size, annotation quality and class balance.
- Detection results should be validated before using the system for workplace safety decisions.
- The current project is a development and demonstration prototype, not a certified safety system.

## 🎓 Seminar Information

**Seminar Framework:** Ultralytics YOLO

**Seminar Title:** Real-Time Object Detection in Industrial Applications

**Project:** IndustrialVision AI

**Domain:** Artificial Intelligence, Machine Learning and Computer Vision

**Core Technology:** YOLO-based Object Detection

## 👩‍💻 Author

**Pallampati Devisri Sarvani**

B.Tech – Artificial Intelligence and Machine Learning

GitHub: [vanisri6396-hue](https://github.com/vanisri6396-hue)

LinkedIn: [Connect on LinkedIn](https://linkedin.com/in/sarvani-pallampati-36391a326/)

## 📜 License

This project is intended for educational, research and demonstration purposes.

The project code may be distributed under the MIT License. Dataset and pretrained model usage remain subject to their respective licenses and terms.

---

⭐ If you find this project interesting, consider giving the repository a star!

**IndustrialVision AI – Towards Smarter and Safer Industrial Environments.**