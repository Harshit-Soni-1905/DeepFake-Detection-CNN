# 🔎 FRAMECHECK | DeepFake Detection CNN

### CNN-based DeepFake Image Detection with PyTorch and Streamlit

<p align="center">
  A deep learning based image classification system for detecting manipulated facial images.
</p>

<p align="center">
  <a href="https://framecheck-deepfake.streamlit.app/">
    🚀 <b>Live Demo</b>
  </a>
</p>

<p align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">

<img src="https://img.shields.io/badge/TorchVision-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="TorchVision">

<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">

<img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">

<img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white" alt="Git">

<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">

</p>

---

## 📌 Overview

**FRAMECHECK** is a deep learning based image classification project that detects whether a facial image is **Real** or **Fake** using a custom Convolutional Neural Network (CNN).

The project covers the complete workflow from image preprocessing and CNN training to model evaluation, single-image inference and deployment through Streamlit.

```text
Image
  ↓
Preprocessing
  ↓
CNN
  ↓
Softmax
  ↓
Fake / Real
```

---

## 🎯 Objective

The objective of this project is to build a CNN-based system capable of learning visual patterns from facial images and classifying them into two categories:

```text
0 → Fake
1 → Real
```

---

## 🗂️ Dataset

The dataset is divided into training, validation and test sets:

| Split | Images |
|---|---:|
| Training | 100,000 |
| Validation | 20,000 |
| Testing | 20,000 |

The model was trained using the training set, the validation set was used for model selection, and the test set was kept for final evaluation.

---

## ⚙️ Preprocessing

Images are resized to **128 × 128** and normalized before being passed to the CNN.

### Training

- Resize to 128 × 128
- Random Horizontal Flip
- Random Rotation up to 10°
- Convert to Tensor
- Normalize

### Validation / Testing

- Resize to 128 × 128
- Convert to Tensor
- Normalize

Normalization:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

## 🧠 CNN Architecture

FRAMECHECK uses a custom CNN implemented using PyTorch.

```text
Input: 3 × 128 × 128
        │
        ▼
Conv2D: 3 → 32
        │
      ReLU
        │
    MaxPool
        │
        ▼
Conv2D: 32 → 64
        │
      ReLU
        │
    MaxPool
        │
        ▼
Conv2D: 64 → 128
        │
      ReLU
        │
    MaxPool
        │
        ▼
Flatten
        │
        ▼
FC: 32768 → 128
        │
      ReLU
        │
   Dropout 0.5
        │
        ▼
FC: 128 → 2
        │
        ▼
   Fake / Real
```

The convolutional layers learn visual features from the images, while the fully connected layers perform the final classification.

---

## 📈 Training

The model was trained for **10 epochs**.

The best validation performance was achieved during Epoch 10:

```text
Validation Accuracy: 91.21%
Training Accuracy:    89.44%
```

The best-performing model was saved as:

```text
best_deepfake_cnn.pth
```

---

## 📊 Results

The final model was evaluated on **20,000 unseen test images**.

| Metric | Score |
|---|---:|
| **Test Accuracy** | **90.85%** |
| **Precision** | **90.87%** |
| **Recall** | **90.82%** |
| **F1 Score** | **90.85%** |

### Confusion Matrix

```text
                  Predicted
              Fake       Real
Actual Fake   9082        918
Actual Real    912       9088
```

The relatively balanced number of false predictions indicates that the model performs similarly across both classes on the held-out test set.

---

## 🔍 Inference Pipeline

For a new uploaded image, FRAMECHECK performs:

```text
Upload Image
     ↓
RGB Conversion
     ↓
Resize 128 × 128
     ↓
Tensor Conversion
     ↓
Normalization
     ↓
CNN
     ↓
Softmax Probabilities
     ↓
Fake / Real Prediction
```

The application also displays the confidence associated with the predicted class.

---

## 🌐 Streamlit Application

The trained model is integrated into a custom Streamlit interface that allows users to:

- Upload JPG, JPEG or PNG images
- Preview the uploaded image
- Run CNN inference
- View Fake / Real classification
- View prediction confidence
- View model and input information

### 🚀 Live Demo

**[FRAMECHECK](https://framecheck-deepfake.streamlit.app/)**

---

## 🛠️ Tech Stack

- **Python**
- **PyTorch**
- **TorchVision**
- **Pillow**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**
- **Streamlit**
- **Git & GitHub**
- **Streamlit Community Cloud**

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

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd DeepFake-Detection-CNN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

The application will be available at:

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

- Train on more diverse deepfake datasets
- Improve generalization to unseen manipulation techniques
- Experiment with pretrained models such as ResNet, EfficientNet and Xception
- Add face detection and cropping
- Implement Grad-CAM for model explainability
- Perform cross-dataset evaluation
- Extend the system to video deepfake detection

---

## 🎓 Key Learnings

This project provided hands-on experience with:

- CNN architecture design
- PyTorch model training
- Image preprocessing and augmentation
- Model checkpointing
- Confusion matrix analysis
- Precision, Recall and F1 Score
- Single-image inference
- Streamlit application development
- GitHub and cloud deployment
- Understanding model generalization and domain shift

---

## 📜 License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Harshit Soni**

B.Tech Computer Science & Engineering

Interested in Artificial Intelligence, Machine Learning and Deep Learning.

---

<p align="center">
  Built with Python • PyTorch • Streamlit
</p>
