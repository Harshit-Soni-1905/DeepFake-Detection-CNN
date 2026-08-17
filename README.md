# 🔎 FRAMECHECK

### DeepFake Detection using Convolutional Neural Networks

<p align="center">
  <strong>A PyTorch-based image classification system for detecting manipulated facial images</strong>
</p>

<p align="center">
  <a href="https://framecheck-deepfake.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-FRAMECHECK-FF4B4B?style=for-the-badge" alt="Live Demo">
  </a>
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit" alt="Streamlit">
</p>

---

## 📌 About the Project

**FRAMECHECK** is a deep learning project that classifies facial images into two categories:

- **Fake**: Manipulated or synthetic facial image
- **Real**: Authentic facial image

The project was developed using a custom **Convolutional Neural Network (CNN)** with PyTorch and deployed as an interactive web application using Streamlit.

Rather than treating deepfake detection as only a model-training problem, this project implements an end-to-end machine learning workflow:

```text
Dataset
   ↓
Image Preprocessing
   ↓
Data Augmentation
   ↓
CNN Training
   ↓
Validation
   ↓
Best Model Checkpoint
   ↓
Test Evaluation
   ↓
Inference Pipeline
   ↓
Streamlit Application
   ↓
Cloud Deployment
