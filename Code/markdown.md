# Traffic Sign Detection

Project ID: PRAICP-1002-TrafSignDetc

## Project Overview

This project implements a Convolutional Neural Network (CNN) to detect and classify traffic signs. The model is trained on a dataset containing 43 different types of traffic signs.

## Table of Contents

- [Traffic Sign Detection](#traffic-sign-detection)
  - [Project Overview](#project-overview)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Dataset Structure](#dataset-structure)
  - [Model Architecture](#model-architecture)
  - [Data Preprocessing](#data-preprocessing)
  - [Training](#training)
  - [Results](#results)
  - [Usage](#usage)
- [Load the model](#load-the-model)
- [Preprocess the image](#preprocess-the-image)
- [Make prediction](#make-prediction)
- [Print predicted class](#print-predicted-class)
  - [Model Performance](#model-performance)

## Requirements

```python
tensorflow
keras
matplotlib
numpy
pandas
opencv-python (cv2)
PIL
sklearn
```

## Dataset Structure

The dataset is organized into training and testing sets:

* Training images: Located in [Train](vscode-file://vscode-app/c:/Users/tamil/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)
* Testing images: Located in [Test](vscode-file://vscode-app/c:/Users/tamil/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)
* Image size: 33x33 pixels
* Number of classes: 43
* Color format: Grayscale (converted from RGB)

## Model Architecture

Model: Sequential

| **Layer (type)**   | **Output Shape** | **Param #** |
| ------------------------ | ---------------------- | ----------------- |
| Conv2D (32 filters, 5x5) | (None, 29, 29, 32)     | 832               |
| Conv2D (32 filters, 3x3) | (None, 27, 27, 32)     | 9,248             |
| MaxPooling2D (2x2)       | (None, 13, 13, 32)     | 0                 |
| Dropout (0.25)           | (None, 13, 13, 32)     | 0                 |
| Conv2D (64 filters, 3x3) | (None, 11, 11, 64)     | 18,496            |
| MaxPooling2D (2x2)       | (None, 5, 5, 64)       | 0                 |
| Dropout (0.25)           | (None, 5, 5, 64)       | 0                 |
| Flatten                  | (None, 1600)           | 0                 |
| Dense                    | (None, 256)            | 409,856           |
| Dropout (0.5)            | (None, 256)            | 0                 |
| Dense                    | (None, 43)             | 11,051            |
| **Total**          |                        | **449,483** |

---

## Data Preprocessing

1. Image resizing to 33x33 pixels
2. Conversion to grayscale
3. Histogram equalization
4. Normalization (dividing by 255.0)
5. Data augmentation using ImageDataGenerator:
   * Width shift: ±10%
   * Height shift: ±10%
   * Zoom range: 20%
   * Shear range: 10%
   * Rotation range: ±10 degrees

## Training

* Optimizer: Adam
* Loss function: Categorical Crossentropy
* Metrics: Accuracy
* Epochs: 15
* Early stopping with patience of 4 epochs
* Batch size: 32

## Results

* Training accuracy: ~98%
* Validation accuracy: ~95%
* The model successfully classifies 43 different types of traffic signs

from tensorflow.keras.models import load_model
import cv2
import numpy as np

## Usage

To predict a traffic sign class:

```python
# Load the model
model = load_model("model_traffic_data.keras")

# Preprocess the image
img = cv2.imread("path_to_image")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (33, 33))
img = img / 255.0
img = np.expand_dims(img, axis=-1)
img = np.expand_dims(img, axis=0)

# Make prediction
prediction = model.predict(img)
predicted_class = np.argmax(prediction, axis=1)

# Print predicted class
print(f"Predicted Class ID: {predicted_class[0]}")
print(f"Predicted Class: {classes[predicted_class[0]]}")
```

Predicted Class ID: 8
Predicted Class: Speed limit (120km/h)

## Model Performance

The model shows good performance in detecting and classifying traffic signs with:

* High accuracy on both training and validation sets
* Good generalization capabilities
* Robust performance against various image conditions
