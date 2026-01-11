# 👁️ Computer Vision Projects

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green.svg)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4%2B-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive collection of **Computer Vision** projects demonstrating expertise in **Image Classification**, **Object Detection**, and **OpenCV Fundamentals** using TensorFlow, Keras, and OpenCV.

---

## 📋 Table of Contents
- [Projects Overview](#-projects-overview)
- [Technologies Used](#️-technologies-used)
- [Installation](#-installation)
- [Project Details](#-project-details)
- [Key CV Concepts](#-key-cv-concepts)
- [Results](#-results)
- [Contact](#-contact)

---

## 🚀 Projects Overview

| # | Project | Category | Notebook | Technique |
|---|---------|----------|----------|-----------|
| 1 | **Chicken Classification** | Deep Learning | [`01_chicken_classification_cnn.ipynb`](01_chicken_classification_cnn.ipynb) | CNN Image Classification |
| 2 | **ANN Image Classification** | Deep Learning | [`02_ann_image_classification.ipynb`](02_ann_image_classification.ipynb) | Artificial Neural Networks |
| 3 | **Flip Operations** | OpenCV Basics | [`03_opencv_flip_operations.ipynb`](03_opencv_flip_operations.ipynb) | Image Transformations |
| 4 | **Image Operations** | OpenCV Basics | [`04_opencv_image_operations.ipynb`](04_opencv_image_operations.ipynb) | Filtering, Blurring, Edge Detection |
| 5 | **Image Pyramids** | OpenCV Advanced | [`05_image_pyramid.ipynb`](05_image_pyramid.ipynb) | Multi-scale Processing |
| 6 | **Hand Detection** | Object Detection | [`06_hand_detection_contours.ipynb`](06_hand_detection_contours.ipynb) | Contour Detection |
| 7 | **Video Shape Insertion** | Video Processing | [`07_video_shape_insertion.ipynb`](07_video_shape_insertion.ipynb) | Real-time Video Manipulation |

---

## 🛠️ Technologies Used

### Core CV Libraries
- **OpenCV** - Computer vision operations
- **TensorFlow/Keras** - Deep learning for CV
- **PIL/Pillow** - Image processing
- **NumPy** - Array operations

### Computer Vision Techniques
- **CNNs** - Convolutional Neural Networks
- **Image Processing** - Filtering, transformations
- **Object Detection** - Contour-based detection
- **Video Processing** - Real-time frame manipulation

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/uzi-gpu/computer-vision-projects.git
   cd computer-vision-projects
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

---

## 📊 Project Details

### Deep Learning Projects

#### 1. 🐔 Chicken Classification with CNN

**File:** [`01_chicken_classification_cnn.ipynb`](01_chicken_classification_cnn.ipynb)

**Objective:** Build a CNN to classify images of chickens into different categories

**Dataset:** Custom chicken image dataset with multiple classes

**Model Architecture:**
- Multiple Conv2D layers with ReLU activation
- MaxPooling for dimension reduction
- Batch normalization for stability
- Fully connected classifier head
- Softmax output

**Key Features:**
- ✅ Data augmentation (rotation, flip, zoom)
- ✅ Custom CNN architecture
- ✅ Training with validation split
- ✅ Model evaluation metrics
- ✅ Confusion matrix analysis
- ✅ Prediction visualization

---

#### 2. 🧠 ANN for Image Classification

**File:** [`02_ann_image_classification.ipynb`](02_ann_image_classification.ipynb)

**Objective:** Comprehensive image classification using Artificial Neural Networks

**Tasks Covered:**
- Multi-class image classification
- Model architecture design
- Hyperparameter tuning
- Performance optimization

**Implementation:**
- ✅ Dense neural network layers
- ✅ Dropout for regul arization
- ✅ Batch normalization
- ✅ Learning rate scheduling
- ✅ Model checkpointing
- ✅ Extensive evaluation

---

### OpenCV Fundamentals

#### 3. 🔄 Image Flip Operations

**File:** [`03_opencv_flip_operations.ipynb`](03_opencv_flip_operations.ipynb)

**Objective:** Master image transformation techniques

**Operations Covered:**
- Horizontal flip (mirror)
- Vertical flip
- Both axes flip (180° rotation)
- Custom flip angles

**Applications:**
- Data augmentation for ML
- Image preprocessing
- Photo editing

---

#### 4. 🎨 Image Operations

**File:** [`04_opencv_image_operations.ipynb`](04_opencv_image_operations.ipynb)

**Objective:** Fundamental image processing operations

**Techniques:**

**1. Filtering:**
- Gaussian blur (noise reduction)
- Median blur (salt-and-pepper noise)
- Bilateral filter (edge-preserving smoothing)

**2. Edge Detection:**
- Canny edge detector
- Sobel operator
- Laplacian edge detection

**3. Morphological Operations:**
- Erosion and dilation
- Opening and closing
- Morphological gradient

**4. Color Space Conversions:**
- RGB to Grayscale
- RGB to HSV
- Color masking

---

#### 5. 🏔️ Image Pyramids

**File:** [`05_image_pyramid.ipynb`](05_image_pyramid.ipynb)

**Objective:** Multi-scale image representation and processing

**Types:**
- **Gaussian Pyramid** - Progressive downsampling
- **Laplacian Pyramid** - Difference of Gaussian

**Applications:**
- ✅ Image blending
- ✅ Multi-scale feature detection
- ✅ Object detection at different scales
- ✅ Image compression

---

### Object Detection & Video Processing

#### 6. ✋ Hand Detection with Contours

**File:** [`06_hand_detection_contours.ipynb`](06_hand_detection_contours.ipynb)

**Objective:** Detect and track hands using contour-based methods

**Pipeline:**
1. **Skin color detection** - HSV color space thresholding
2. **Contour extraction** - Find hand boundaries
3. **Contour filtering** - Remove noise
4. **Hand localization** - Bounding box detection

**Techniques:**
- ✅ Color space conversion
- ✅ Morphological operations
- ✅ Contour detection (`cv2.findContours`)
- ✅ Convex hull
- ✅ Bounding rectangle

---

#### 7. 🎬 Video Shape Insertion

**File:** [`07_video_shape_insertion.ipynb`](07_video_shape_insertion.ipynb)

**Objective:** Real-time video processing and shape overlay

**Operations:**
- Drawing rectangles
- Drawing circles
- Drawing lines
- Adding text overlays
- Real-time frame manipulation

**Applications:**
- Video annotation
- Object tracking visualization
- AR overlays
- Video effects

---

## 📚 Key CV Concepts Demonstrated

### Image Processing Fundamentals
1. **Color Spaces** - RGB, HSV, Grayscale conversions
2. **Filtering** - Smoothing, sharpening, noise reduction
3. **Edge Detection** - Finding object boundaries
4. **Morphological Operations** - Shape manipulation

### Deep Learning for CV
1. **CNNs** - Convolutional architectures
2. **Data Augmentation** - Increasing dataset diversity
3. **Transfer Learning** - Pre-trained models
4. **Model Optimization** - Hyperparameter tuning

### Object Detection
1. **Contour Detection** - Shape finding
2. **Color-based Detection** - HSV thresholding
3. **Feature Extraction** - Hand-crafted features
4. **Bounding Boxes** - Object localization

### Video Processing
1. **Frame Extraction** - Video to image sequences
2. **Real-time Processing** - Efficient algorithms
3. **Drawing Operations** - Annotations
4. **Video I/O** - Reading and writing videos

---

## 🏆 Results

### Chicken Classification CNN
- **Training Accuracy:** High performance on training set
- **Validation Accuracy:** Good generalization
- **Data Augmentation:** Improved robustness
- **Model:** Production-ready classifier

### ANN Image Classification
- **Architecture:** Optimized neural network
- **Performance:** Competitive accuracy
- **Insights:** Deep learning effectiveness

### OpenCV Operations
- **Processing Speed:** Real-time capable
- **Quality:** Professional-grade results
- **Applications:** Ready for production use

### Hand Detection
- **Detection Rate:** Reliable hand localization
- **Robustness:** Works in various lighting
- **Performance:** Real-time processing

---

## 🎓 Learning Outcomes

Through these projects, I have demonstrated proficiency in:

1. **Computer Vision Foundations**
   - Image processing basics
   - OpenCV operations
   - Color space manipulations
   - Filter applications

2. **Deep Learning for CV**
   - CNN architecture design
   - Image classification pipelines
   - Model training and evaluation
   - Data augmentation strategies

3. **Object Detection**
   - Contour-based detection
   - Color thresholding
   - Morphological operations
   - Real object localization

4. **Practical Applications**
   - Real-time video processing
   - Multi-scale image analysis
   - Production-ready implementations
   - Performance optimization

---

## 📧 Contact

**Uzair Mubasher** - BSAI Graduate

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/uzair-mubasher-208ba5164)
[![Email](https://img.shields.io/badge/Email-uzairmubasher5@gmail.com-red)](mailto:uzairmubasher5@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-uzi--gpu-black)](https://github.com/uzi-gpu)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- OpenCV community and documentation
- TensorFlow/Keras teams
- Computer Vision course instructors

---

**⭐ If you found this repository helpful, please consider giving it a star!**
