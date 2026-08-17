# 🔎 FRAMECHECK

### DeepFake Detection using a Custom CNN

<p align="center">
  <b>Detect whether a facial image is Real or Fake using a PyTorch-based deep learning model.</b>
</p>

<p align="center">
  <a href="https://framecheck-deepfake.streamlit.app/">
    🚀 <b>Live Demo</b>
  </a>
</p>

<p align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">

<img src="https://img.shields.io/badge/TorchVision-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">

<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">

<img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">

<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white">

</p>

---

## 🚀 Results at a Glance

| Metric | Result |
|---|---:|
| **Test Accuracy** | **90.85%** |
| **Precision** | **90.87%** |
| **Recall** | **90.82%** |
| **F1 Score** | **90.85%** |
| **Training Images** | **100,000** |
| **Validation Images** | **20,000** |
| **Test Images** | **20,000** |

> **[Try FRAMECHECK Live →](https://framecheck-deepfake.streamlit.app/)**

---

## 📌 About

**FRAMECHECK** is an end-to-end deep learning project for **image-based deepfake detection**.

A custom Convolutional Neural Network was trained using **140,000 images** across training, validation and test sets. The model learns visual patterns from facial images and performs binary classification:

```text
Input Image
     ↓
Image Preprocessing
     ↓
Custom CNN
     ↓
Class Probabilities
     ↓
┌─────────────┐
│ Fake / Real │
└─────────────┘
```

The trained model was then integrated into a Streamlit application and deployed as a publicly accessible web app.

---

## 🧠 Model Architecture

The model is a custom CNN built using PyTorch rather than relying on a pretrained architecture.

```text
Input: 3 × 128 × 128
          │
          ▼
     Conv2D 3 → 32
          │
        ReLU
          │
       MaxPool
          │
          ▼
     Conv2D 32 → 64
          │
        ReLU
          │
       MaxPool
          │
          ▼
     Conv2D 64 → 128
          │
        ReLU
          │
       MaxPool
          │
          ▼
        Flatten
          │
          ▼
     FC 32768 → 128
          │
        ReLU
          │
      Dropout 0.5
          │
          ▼
      FC 128 → 2
          │
          ▼
      Fake / Real
```

### Key Design Choices

- **3 convolution blocks** for hierarchical feature extraction
- **ReLU** for non-linear feature learning
- **MaxPooling** for spatial downsampling
- **Dropout (0.5)** to reduce overfitting
- **Fully connected layers** for final classification

---

## 🖼️ Data Processing

Images are resized to **128 × 128** before being passed to the model.

### Training Transformations

```text
Resize
  ↓
Random Horizontal Flip
  ↓
Random Rotation (10°)
  ↓
ToTensor
  ↓
Normalize
```

### Validation / Test Transformations

```text
Resize
  ↓
ToTensor
  ↓
Normalize
```

Normalization:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Using separate training and evaluation transformations ensures that augmentation is applied only during training while validation and test evaluation remain deterministic.

---

## 🏋️ Training

The model was trained for **10 epochs**.

The best validation performance was obtained at Epoch 10:

```text
Training Accuracy   : 89.44%
Validation Accuracy : 91.21%
```

The best-performing checkpoint was saved as:

```text
best_deepfake_cnn.pth
```

The checkpoint is used directly by the deployed application, so the model does not need to be retrained during inference.

---

## 📊 Evaluation

The final checkpoint was evaluated on **20,000 unseen test images**.

### Performance

| Metric | Score |
|---|---:|
| Accuracy | **90.85%** |
| Precision | **90.87%** |
| Recall | **90.82%** |
| F1 Score | **90.85%** |

### Confusion Matrix

```text
                  Predicted
              Fake       Real
Actual Fake   9082        918
Actual Real    912       9088
```

The relatively balanced false-positive and false-negative counts indicate that the model performs consistently across both classes on the held-out test set.

---

## 🔍 Inference Pipeline

When an image is uploaded to FRAMECHECK:

```text
             Uploaded Image
                    │
                    ▼
              RGB Conversion
                    │
                    ▼
              Resize 128×128
                    │
                    ▼
              Tensor Conversion
                    │
                    ▼
                Normalize
                    │
                    ▼
                 CNN Model
                    │
                    ▼
                  Logits
                    │
                    ▼
                 Softmax
                    │
                    ▼
             Class Probabilities
                    │
                    ▼
             Fake / Real + Confidence
```

The inference pipeline uses the same preprocessing strategy as validation and testing.

---

## 🌐 Web Application

The trained model is exposed through a custom Streamlit interface.

### Application Features

- 📤 Image upload
- 🖼️ Image preview
- 🧠 CNN-based prediction
- 📊 Confidence score
- ⚡ CPU-based inference
- 🎨 Custom interface
- ☁️ Public cloud deployment

### Live Demo

**[🔎 Open FRAMECHECK](https://framecheck-deepfake.streamlit.app/)**

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Development |
| **PyTorch** | CNN architecture and inference |
| **TorchVision** | Image transformations |
| **Pillow** | Image processing |
| **Scikit-learn** | Evaluation metrics |
| **Matplotlib / Seaborn** | Visualization |
| **Streamlit** | Web interface |
| **Git / GitHub** | Version control |
| **Streamlit Cloud** | Deployment |

---

## 📁 Project Structure

```text
DeepFake-Detection-CNN/
│
├── app.py
├── best_deepfake_cnn.pth
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 💻 Run Locally

### Clone

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd DeepFake-Detection-CNN
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## ⚠️ Generalization & Limitations

The model achieves **90.85% accuracy on the held-out test set**.

During additional experimentation with externally sourced internet images, the model showed weaker generalization than on images from the original dataset.

This highlights an important real-world challenge in deepfake detection: **distribution shift**.

Different sources can introduce variations in:

- Image compression
- Resolution
- Facial characteristics
- Manipulation techniques
- Generation pipelines

Therefore, the current model is best viewed as a **deepfake detection prototype** evaluated primarily within the distribution represented by its dataset.

This also provides a clear direction for future improvement rather than hiding the model's limitations.

---

## 🔮 Future Improvements

- Train on multiple and more diverse deepfake datasets
- Improve cross-dataset generalization
- Experiment with ResNet, EfficientNet and Xception
- Add dedicated face detection and cropping
- Integrate Grad-CAM for explainability
- Perform external dataset evaluation
- Extend from image detection to video-level detection

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Harshit Soni**

B.Tech Computer Science & Engineering

**Interests:** AI/ML • Deep Learning • Computer Vision

---

<p align="center">
  <b>FRAMECHECK</b><br>
  Built with Python • PyTorch • Streamlit
</p>
