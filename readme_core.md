# 🏭 IndustrialVision AI
### Real-Time Object Detection in Industrial Applications using YOLO

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-purple)
![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-orange)
![Status](https://img.shields.io/badge/Status-Development-yellow)

## 📌 Project Overview

**IndustrialVision AI** is a computer vision project designed to demonstrate real-time object detection and tracking in industrial environments using the YOLO (You Only Look Once) object detection architecture.

The project initially explores pretrained YOLO models using the COCO dataset and extends the implementation through custom dataset preparation and model fine-tuning for industrial Personal Protective Equipment (PPE) detection.

The system aims to identify safety-related objects in industrial environments, helping demonstrate how AI-powered computer vision can support workplace safety monitoring.

## 🎯 Objectives

- Understand and implement YOLO-based object detection.
- Perform object detection on images and videos.
- Detect multiple objects with bounding boxes, class labels, and confidence scores.
- Implement object tracking with tracking IDs.
- Prepare and validate a custom industrial PPE dataset.
- Fine-tune a pretrained YOLO model on custom classes.
- Evaluate model performance using standard object detection metrics.
- Explore real-world applications of AI in industrial safety.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Ultralytics YOLO | Object detection and training |
| OpenCV | Image and video processing |
| PyTorch | Deep learning framework |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Google Colab | Model training |
| VS Code | Development environment |
| Git & GitHub | Version control |

## 🧠 Project Workflow

1. **Pretrained Model Implementation:** Load pretrained YOLO weights and perform initial object detection using the COCO-trained model.
2. **Image Detection:** Detect objects in static images and visualize bounding boxes, labels, and confidence scores.
3. **Video Detection:** Process video frames to perform continuous object detection.
4. **Object Tracking:** Track detected objects across video frames and assign tracking IDs.
5. **Custom Dataset Preparation:** Prepare an industrial PPE dataset containing images and YOLO-format annotation files.
6. **Dataset Validation:** Verify image-label correspondence, missing annotations, and class distribution.
7. **Model Fine-Tuning:** Fine-tune pretrained YOLO weights using the custom PPE dataset.
8. **Model Evaluation:** Evaluate the trained model using Precision, Recall, mAP50, and mAP50-95.
9. **Application Development:** Integrate the trained model into an application for industrial object detection.

## 📂 Project Structure

```text
IndustrialVision_AI/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── image_detection.py
├── video_detection.py
├── tracking.py
├── validate_dataset.py
│
├── train.py
├── evaluate.py
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
└── runs/
    └── detect/
        └── train/
```

*Note: The structure above is illustrative. Update filenames and folders to match your actual project. Large datasets, generated outputs, and model weights may be maintained separately rather than committed to GitHub.*

## 📊 Dataset Information

The project uses two dataset stages:

### 1. COCO Pretrained Dataset

The initial object detection experiments use pretrained YOLO weights trained on the COCO dataset.

COCO provides a broad range of everyday object categories and helps demonstrate general-purpose object detection.

### 2. Custom Construction PPE Dataset

A custom construction safety dataset is prepared for fine-tuning the pretrained model.

**Dataset classes:**

- Helmet
- Vest
- Other classes as defined in the custom dataset configuration.

**Dataset preparation:**

| Split | Images | Labels |
|---|---:|---:|
| Training | 800 | 800 |
| Validation | 200 | 200 |
| **Total** | **1,000** | **1,000** |

Dataset validation confirmed that each training and validation image had a corresponding label file, with no missing or extra label files.

## ⚙️ Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/IndustrialVision_AI.git
```

### Step 2: Navigate to the project

```bash
cd IndustrialVision_AI
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Running the Project

### Image Detection

```bash
python image_detection.py
```

Detects objects in an input image and saves the annotated output.

### Video Detection

```bash
python video_detection.py
```

Processes video frames and generates an annotated video.

### Object Tracking

```bash
python tracking.py
```

Performs object tracking and assigns tracking IDs to detected objects.

### Dataset Validation

```bash
python validate_dataset.py
```

Checks image-label correspondence and dataset class distribution.

### Model Training

```bash
python train.py
```

*Training commands and dataset paths should match your actual training script and configuration.*

## 📈 Model Evaluation

The initial pretrained COCO-model experiment produced the following evaluation results:

| Metric | Score |
|---|---:|
| Precision | 0.7940 |
| Recall | 0.8667 |
| mAP@50 | 0.9170 |
| mAP@50-95 | 0.6933 |

### Performance Metrics Explained

- **Precision:** Measures how many predicted positive detections are correct.
- **Recall:** Measures how many actual objects are successfully detected.
- **mAP@50:** Mean Average Precision at an IoU threshold of 0.50.
- **mAP@50-95:** Average mAP across IoU thresholds from 0.50 to 0.95.

**Note:** These results belong to the initial evaluation experiment and should not be interpreted as the final performance of the custom PPE fine-tuned model. Update this section with the custom model's results after training and validation.

## 🔍 Key Concepts

### 1. YOLO (You Only Look Once)
A real-time object detection approach that predicts object locations and classes in a single forward pass.

### 2. Bounding Boxes
Rectangular boxes used to locate detected objects in an image.

### 3. Confidence Score
Indicates the model's confidence in a particular detection.

### 4. IoU (Intersection over Union)
Measures the overlap between predicted and ground-truth bounding boxes.

### 5. NMS (Non-Maximum Suppression)
Removes redundant overlapping detections by retaining the most confident predictions according to the configured IoU threshold.

### 6. Fine-Tuning
Adapts a pretrained model to a specific task using a custom labeled dataset.

### 7. Object Tracking
Maintains object identities across consecutive video frames.

## 🏭 Real-World Applications

- Industrial workplace safety monitoring.
- Helmet and safety vest detection.
- Construction site surveillance.
- Factory monitoring.
- Automated visual inspection.
- AI-assisted safety compliance systems.

## 🔮 Future Enhancements

- Real-time CCTV camera integration.
- Advanced PPE compliance monitoring.
- Detection of workers without safety equipment.
- Alert and notification system for safety violations.
- Streamlit or web-based monitoring dashboard.
- Multiple camera support.
- Database integration for incident records.
- Performance optimization for edge devices.
- Integration with additional AI models and analytics.

## 👩‍💻 Author

**Devi Sri Sarvani Pallampati**

B.Tech – Artificial Intelligence and Machine Learning

GitHub: https://github.com/vanisri6396-hue

LinkedIn: https://linkedin.com/in/sarvani-pallampati-36391a326/

## 📜 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) – YOLO framework.
- [COCO Dataset](https://cocodataset.org/) – General-purpose object detection dataset.
- [OpenCV](https://opencv.org/) – Computer vision library.
- [PyTorch](https://pytorch.org/) – Deep learning framework.

## 📄 License

This project is intended for educational and research purposes. Dataset usage and redistribution are subject to their respective licenses. A project license can be added separately.

---

⭐ If you find this project interesting, consider giving the repository a star!