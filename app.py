import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class DeepfakeCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolution Layers
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.relu = nn.ReLU()

        # Fully Connected Layer
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        # First convolution block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Second convolution block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Third convolution block
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully Connected Layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output Layer
        x = self.fc2(x)

        return x

# Image Transformation

valid_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load Trained Model

device = torch.device("cpu")

model = DeepfakeCNN()

model.load_state_dict(
    torch.load(
        "best_deepfake_cnn.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Model loaded successfully")


# Streamlit Interface

st.set_page_config(
    page_title="FrameCheck | Image Authenticity",
    page_icon="🔎",
    layout="wide"
)

st.markdown("""
<style>

    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Animated gradient background instead of flat black */
    .stApp {
        background: radial-gradient(circle at 20% 20%, #1f1147 0%, #0f1220 45%, #0a0c16 100%);
        background-attachment: fixed;
        color: #eaeaf5;
    }

    /* soft glowing blobs floating in the background for depth */
    .stApp::before {
        content: "";
        position: fixed;
        top: -150px;
        right: -150px;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(108,99,255,0.35) 0%, rgba(108,99,255,0) 70%);
        filter: blur(20px);
        z-index: 0;
        pointer-events: none;
    }

    .stApp::after {
        content: "";
        position: fixed;
        bottom: -200px;
        left: -150px;
        width: 550px;
        height: 550px;
        background: radial-gradient(circle, rgba(0,212,255,0.25) 0%, rgba(0,212,255,0) 70%);
        filter: blur(20px);
        z-index: 0;
        pointer-events: none;
    }

    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 52px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 0px;
        background: linear-gradient(90deg, #6C63FF 0%, #00D4FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }

    .subtitle {
        color: #9aa0b4;
        font-size: 16px;
        margin-top: 6px;
        margin-bottom: 40px;
        letter-spacing: 0.5px;
    }

    .section-label {
        color: #8f96b3;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    /* Glassmorphism card */
    .info-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 16px;
        padding: 26px;
        margin-bottom: 18px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(108, 99, 255, 0.2);
    }

    .status-title {
        color: #8f96b3;
        font-size: 12px;
        margin-bottom: 10px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .status-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 34px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .status-fake {
        color: #ff5c7a;
        text-shadow: 0 0 20px rgba(255, 92, 122, 0.45);
    }

    .status-real {
        color: #38f2a3;
        text-shadow: 0 0 20px rgba(56, 242, 163, 0.45);
    }

    .model-info {
        color: #8f96b3;
        font-size: 13px;
        padding-top: 16px;
        margin-top: 14px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        line-height: 1.9;
    }

    .model-info b {
        color: #eaeaf5;
    }

    /* File uploader styling */
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1.5px dashed rgba(108, 99, 255, 0.45);
        border-radius: 16px;
        padding: 14px;
        transition: border-color 0.25s ease, background 0.25s ease;
    }

    div[data-testid="stFileUploader"]:hover {
        border-color: #6C63FF;
        background: rgba(108, 99, 255, 0.06);
    }

    /* Uploaded image gets rounded corners + glow border */
    div[data-testid="stImage"] img {
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 28px rgba(0,0,0,0.4);
    }

    /* Confidence bar */
    .confidence-bar-bg {
        width: 100%;
        height: 8px;
        background: rgba(255,255,255,0.08);
        border-radius: 6px;
        margin-top: 10px;
        overflow: hidden;
    }

    .confidence-bar-fill {
        height: 100%;
        border-radius: 6px;
    }

</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="main-title">FRAMECHECK</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Image Authenticity Analysis · Deepfake Detection</div>',
    unsafe_allow_html=True
)


left_column, right_column = st.columns([1.15, 0.85], gap="large")


with left_column:

    st.markdown(
        '<div class="section-label">Image Input</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="info-card">',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Drop an image here or browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )




if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with left_column:
        st.image(
            image,
            caption="Uploaded image",
            width=500
        )

    image_tensor = valid_test_transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    predicted_class = probabilities.argmax(dim=1).item()
    confidence = probabilities.max().item()

    class_names = ["Fake", "Real"]
    prediction = class_names[predicted_class]

    # Pick color + bar color based on prediction so result stands out
    status_class = "status-fake" if prediction == "Fake" else "status-real"
    bar_color = "#ff5c7a" if prediction == "Fake" else "#38f2a3"
    confidence_pct = confidence * 100

    with right_column:

        st.markdown(
            '<div class="section-label">Analysis</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="info-card">
                <div class="status-title">Classification</div>
                <div class="status-text {status_class}">{prediction.upper()}</div>

                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{confidence_pct:.1f}%; background:{bar_color};"></div>
                </div>

                <div class="model-info">
                    <b>Confidence</b> &nbsp;·&nbsp; {confidence_pct:.2f}%<br>
                    <b>Model</b> &nbsp;·&nbsp; Custom CNN (3 Conv blocks)<br>
                    <b>Input Resolution</b> &nbsp;·&nbsp; 128 × 128
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
