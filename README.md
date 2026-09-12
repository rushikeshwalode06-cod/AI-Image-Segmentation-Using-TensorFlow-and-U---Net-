# AI-Image-Segmentation-Using-TensorFlow-and-U---Net-
Built an AI-based image segmentation system using TensorFlow and a U-Net deep learning architecture. The project uses the Oxford-IIIT Pet dataset, performs image preprocessing and binary mask generation, trains with Dice + Binary Cross-Entropy loss, and evaluates segmentation using Dice and IoU metrics with real-time prediction and visualization.

# 🐾 AI Image Segmentation Using TensorFlow & U-Net

An AI-based image segmentation project that uses **TensorFlow** and a custom **U-Net deep learning architecture** to perform pixel-level image segmentation on the **Oxford-IIIT Pet Dataset**.

The model identifies the foreground object from an image and generates a binary segmentation mask separating the **pet/object from the background**.

---

## 📌 Overview

Image segmentation is a fundamental Computer Vision task where each pixel of an image is assigned to a specific class.

In this project, a **U-Net architecture** is implemented using TensorFlow to perform binary image segmentation. The model is trained on the **Oxford-IIIT Pet Dataset**, where pet images and their corresponding segmentation masks are used for training and evaluation.

The complete pipeline includes:

📂 Dataset Loading
🧹 Image & Mask Preprocessing
🔄 Image Resizing & Normalization
🎭 Binary Mask Generation
🏗️ U-Net Model Construction
🧠 Model Training
📊 Model Evaluation
🔍 Image Segmentation Prediction
📈 Visualization of Results

---

## 🎯 Objective

The main objective of this project is to build a deep learning-based image segmentation system capable of:

🎯 Detecting the foreground pet/object in an image
🎯 Generating pixel-level segmentation masks
🎯 Learning spatial features using U-Net
🎯 Evaluating segmentation performance using Dice and IoU metrics
🎯 Visualizing predicted segmentation results

---

## 📊 Dataset

### Oxford-IIIT Pet Dataset

The project uses the **Oxford-IIIT Pet Dataset** through TensorFlow Datasets (TFDS).

The dataset contains:

🐶 Dog images
🐱 Cat images
🎭 Corresponding segmentation masks

The original segmentation masks are processed and converted into **binary masks** for foreground/background segmentation.

---

## 🧹 Data Preprocessing

The following preprocessing steps are applied before training the model:

🔹 Images are resized to **128 × 128** pixels
🔹 Pixel values are normalized between **0 and 1**
🔹 Segmentation masks are converted into binary masks
🔹 Mask resizing uses **nearest-neighbor interpolation** to preserve class labels
🔹 Data is prepared in batches for efficient model training

### ⚙️ Batch Configuration

```text
Image Size: 128 × 128
Batch Size: 16
Normalization: 0–1
Segmentation Type: Binary
```

---

## 🧠 U-Net Architecture

The project uses a custom **U-Net architecture**, which is widely used for image segmentation tasks.

U-Net follows an **Encoder–Decoder architecture**.

### 🔽 Encoder

The encoder extracts important spatial and semantic features from the input image.

The encoder contains:

🧩 Convolutional Layers
⚡ ReLU Activation
📊 Batch Normalization
⬇️ Max Pooling

The feature channels increase progressively:

```text
32 → 64 → 128 → 256 → 512
```

### 🔼 Decoder

The decoder reconstructs the segmentation map from the extracted features.

It uses:

🔹 Transposed Convolution
🔹 Skip Connections
🔹 Convolutional Layers
🔹 Batch Normalization

Skip connections help preserve important spatial information lost during downsampling.

### 🎭 Output Layer

The final layer uses a **Sigmoid activation function** to generate a pixel-level probability map for binary segmentation.

```text
Output → Segmentation Probability Map
```

---

## 📉 Loss Function

The model is trained using a combination of:

### Binary Cross-Entropy Loss

Measures the difference between predicted probabilities and actual binary segmentation masks.

### Dice Loss

Helps improve segmentation overlap between the predicted mask and ground-truth mask.

The combined loss can be represented as:

```text
Total Loss = Binary Cross-Entropy + Dice Loss
```

This combination helps the model learn both pixel-level classification and segmentation overlap.

---

## 📏 Evaluation Metrics

The model performance is evaluated using segmentation-specific metrics.

### 🎯 Dice Coefficient

Measures the overlap between the predicted segmentation and the ground-truth mask.

Higher Dice score indicates better segmentation performance.

### 🎯 Intersection over Union (IoU)

Measures the intersection between predicted and actual masks relative to their union.

```text
IoU = Intersection / Union
```

### 📊 Validation Loss

Validation loss is monitored during training to evaluate how well the model generalizes to unseen data.

---

## ⚙️ Training

The model training pipeline includes:

🧠 U-Net model training
📉 Binary Cross-Entropy + Dice Loss
📊 Dice and IoU metrics
💾 Model checkpointing
⏹️ Early stopping
📉 Learning rate reduction

### Training Callbacks

The project uses:

🔹 **ModelCheckpoint** – Saves the best-performing model
🔹 **EarlyStopping** – Stops training when validation performance stops improving
🔹 **ReduceLROnPlateau** – Reduces learning rate when validation performance plateaus

---

## 🔍 Prediction & Segmentation

After training, the model can generate segmentation predictions for new images.

The prediction process includes:

📷 Input Image
⬇️
🔄 Resize & Normalize
⬇️
🧠 U-Net Prediction
⬇️
🎭 Probability Map
⬇️
✂️ Threshold at 0.5
⬇️
🖼️ Final Binary Segmentation Mask

The threshold value used for converting the probability map into a binary mask is:

```text
Threshold = 0.5
```

---

## 📈 Visualization

The segmentation results are visualized using multiple outputs:

| Visualization      | Description                               |
| ------------------ | ----------------------------------------- |
| 🖼️ Original Image | Input image provided to the model         |
| 🎭 Ground Truth    | Actual segmentation mask                  |
| 🔮 Prediction      | Model-generated segmentation              |
| 🌈 Overlay         | Segmentation mask over the original image |

These visualizations help compare the model prediction with the actual ground-truth segmentation.

---

## 🛠️ Technology Stack

### 🐍 Programming Language

* **Python**

### 🤖 Deep Learning

* **TensorFlow**
* **U-Net**
* **Convolutional Neural Networks (CNN)**

### 🖼️ Computer Vision

* **Image Segmentation**
* **Binary Mask Generation**
* **Image Preprocessing**

### 📊 Dataset & Data Processing

* **TensorFlow Datasets (TFDS)**
* **NumPy**

### 📈 Visualization

* **Matplotlib**

---

## ⭐ Key Highlights

🚀 Built an end-to-end image segmentation pipeline using TensorFlow
🧠 Implemented a custom U-Net architecture for pixel-level segmentation
🐾 Used the Oxford-IIIT Pet Dataset for training and evaluation
🎭 Converted segmentation annotations into binary masks
📉 Used combined Binary Cross-Entropy and Dice Loss
📊 Evaluated the model using Dice and IoU metrics
🔍 Generated segmentation predictions using a 0.5 threshold
📈 Visualized original images, ground truth masks, predictions, and overlays
⚡ Implemented training callbacks for better model training

---

## 💡 Applications

This type of image segmentation system can be useful in:

🐾 Animal/Object Segmentation
🏥 Medical Image Segmentation
🛰️ Satellite Image Analysis
🚗 Autonomous Driving
🌾 Agricultural Image Analysis
🔬 Scientific Image Processing
👁️ Object Detection & Analysis

---

## 🎯 Conclusion

This project demonstrates how **TensorFlow and U-Net** can be used to build an effective image segmentation system.

By combining image preprocessing, binary mask generation, U-Net architecture, Dice + Binary Cross-Entropy loss, and segmentation evaluation metrics such as **Dice and IoU**, the project provides a complete deep learning workflow for pixel-level image segmentation.

The project also demonstrates how segmentation predictions can be visualized and analyzed against ground-truth masks.

---

⭐ **If you find this project useful, consider giving it a star!**
