# 🔎 FRAMECHECK | DeepFake Detection CNN

### Image Authenticity Analysis using Convolutional Neural Networks

FRAMECHECK is a deep learning based image classification system designed to classify facial images as **Real** or **Fake** using a custom Convolutional Neural Network (CNN).

The project covers the complete machine learning workflow, from image preprocessing and CNN training to model evaluation and deployment as an interactive Streamlit web application.

<p align="center">

<a href="https://framecheck-deepfake.streamlit.app/">
  <img src="https://img.shields.io/badge/🚀%20Live%20Demo-FRAMECHECK-FF4B4B?style=for-the-badge" alt="Live Demo">
</a>

</p>

---

## 📌 Overview

The increasing availability of AI-based face manipulation techniques has made it increasingly difficult to distinguish authentic images from manipulated ones.

FRAMECHECK explores a practical deep learning approach to this problem by training a CNN to learn visual patterns that differentiate between real and fake facial images.

The trained model is evaluated on a held-out test set and then integrated into a Streamlit application for real-time image classification.

---

## ✨ Features

- 🧠 Custom CNN architecture built with PyTorch
- 🖼️ Image classification into **Fake** or **Real**
- 🔄 Consistent preprocessing during training and inference
- 📊 Evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
- ⚡ CPU-based inference for deployment
- 🌐 Interactive Streamlit web application
- ☁️ Deployed using Streamlit Community Cloud

---

## 🏗️ System Architecture

```text
                    INPUT IMAGE
                         │
                         ▼
              ┌─────────────────────┐
              │ Image Preprocessing │
              │ Resize → 128 × 128  │
              │ Normalize           │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │      CNN Model      │
              │                     │
              │ Conv2D → ReLU       │
              │ MaxPool             │
              │ Conv2D → ReLU       │
              │ MaxPool             │
              │ Conv2D → ReLU       │
              │ MaxPool             │
              └──────────┬──────────┘
                         │
                         ▼
                  Fully Connected
                         │
                         ▼
                   Output Layer
                  ┌──────┴──────┐
                  │             │
                FAKE           REAL
