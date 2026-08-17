import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# Deepfake CNN Model

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

        # Fully Connected Layers
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


# Streamlit Page Configuration

st.set_page_config(
    page_title="FRAMECHECK | Image Authenticity",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Custom CSS

st.markdown("""
<style>

    /* Global App Background */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(37, 99, 235, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(124, 58, 237, 0.10),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #070b12 0%,
                #0b111c 50%,
                #080d15 100%
            );

        color: #f8fafc;
    }


    /* Main Container */

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }


    /* Header */

    .brand-container {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 6px;
    }

    .brand-icon {
        width: 48px;
        height: 48px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 14px;

        background:
            linear-gradient(
                135deg,
                rgba(59, 130, 246, 0.25),
                rgba(124, 58, 237, 0.20)
            );

        border: 1px solid rgba(96, 165, 250, 0.25);

        box-shadow:
            0 0 30px rgba(59, 130, 246, 0.12);

        font-size: 24px;
    }

    .brand-name {
        font-size: 34px;
        font-weight: 800;
        letter-spacing: 2px;

        background:
            linear-gradient(
                90deg,
                #ffffff,
                #93c5fd,
                #c4b5fd
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .brand-tagline {
        color: #94a3b8;
        font-size: 13px;
        letter-spacing: 2px;
        margin-left: 62px;
        margin-top: -4px;
        margin-bottom: 30px;
        text-transform: uppercase;
    }


    /* Section Labels */

    .section-label {
        font-size: 11px;
        font-weight: 700;

        color: #64748b;

        letter-spacing: 2px;

        text-transform: uppercase;

        margin-bottom: 12px;
    }


    /* Glass Cards */

    .glass-card {
        background:
            linear-gradient(
                145deg,
                rgba(20, 30, 45, 0.78),
                rgba(10, 17, 28, 0.88)
            );

        border: 1px solid rgba(148, 163, 184, 0.12);

        border-radius: 20px;

        padding: 25px;

        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.30),
            inset 0 1px 0 rgba(255, 255, 255, 0.03);

        backdrop-filter: blur(18px);

        margin-bottom: 20px;
    }


    /* Upload Area */

    .upload-title {
        font-size: 19px;
        font-weight: 700;

        color: #f8fafc;

        margin-bottom: 6px;
    }

    .upload-description {
        font-size: 13px;

        color: #64748b;

        margin-bottom: 20px;
    }

    div[data-testid="stFileUploader"] {
        background:
            linear-gradient(
                145deg,
                rgba(15, 23, 42, 0.85),
                rgba(15, 23, 42, 0.55)
            );

        border: 1px dashed rgba(96, 165, 250, 0.30);

        border-radius: 16px;

        padding: 14px;

        transition: all 0.25s ease;
    }

    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(96, 165, 250, 0.65);

        box-shadow:
            0 0 25px rgba(59, 130, 246, 0.08);
    }


    /* File Uploader Button */

    div[data-testid="stFileUploader"] button {
        border-radius: 10px !important;

        border: 1px solid rgba(96, 165, 250, 0.25) !important;

        background: rgba(30, 41, 59, 0.8) !important;

        color: #e2e8f0 !important;
    }


    /* Result Card */

    .result-card {
        background:
            linear-gradient(
                145deg,
                rgba(15, 23, 42, 0.92),
                rgba(10, 16, 27, 0.96)
            );

        border-radius: 20px;

        padding: 28px;

        border: 1px solid rgba(148, 163, 184, 0.12);

        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.30);
    }

    .result-label {
        color: #64748b;

        font-size: 11px;

        font-weight: 700;

        letter-spacing: 2px;

        text-transform: uppercase;

        margin-bottom: 12px;
    }

    .result-real {
        font-size: 42px;

        font-weight: 800;

        color: #4ade80;

        letter-spacing: 1px;

        text-shadow:
            0 0 25px rgba(74, 222, 128, 0.18);
    }

    .result-fake {
        font-size: 42px;

        font-weight: 800;

        color: #f87171;

        letter-spacing: 1px;

        text-shadow:
            0 0 25px rgba(248, 113, 113, 0.18);
    }


    /* Confidence Section */

    .confidence-title {
        display: flex;

        justify-content: space-between;

        align-items: center;

        color: #94a3b8;

        font-size: 13px;

        margin-top: 28px;

        margin-bottom: 9px;
    }

    .confidence-value {
        color: #f8fafc;

        font-size: 14px;

        font-weight: 700;
    }

    .confidence-track {
        width: 100%;

        height: 7px;

        border-radius: 10px;

        background: #1e293b;

        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;

        border-radius: 10px;

        background:
            linear-gradient(
                90deg,
                #3b82f6,
                #8b5cf6
            );

        box-shadow:
            0 0 14px rgba(99, 102, 241, 0.45);
    }


    /* Model Details */

    .details-grid {
        display: grid;

        grid-template-columns:
            1fr 1fr;

        gap: 10px;

        margin-top: 25px;

        padding-top: 20px;

        border-top:
            1px solid rgba(148, 163, 184, 0.10);
    }

    .detail-item {
        background: rgba(15, 23, 42, 0.60);

        border-radius: 12px;

        padding: 13px;
    }

    .detail-title {
        color: #64748b;

        font-size: 10px;

        text-transform: uppercase;

        letter-spacing: 1px;

        margin-bottom: 4px;
    }

    .detail-value {
        color: #e2e8f0;

        font-size: 13px;

        font-weight: 600;
    }


    /* Waiting State */

    .waiting-card {
        min-height: 360px;

        display: flex;

        flex-direction: column;

        align-items: center;

        justify-content: center;

        text-align: center;

        background:
            linear-gradient(
                145deg,
                rgba(15, 23, 42, 0.65),
                rgba(10, 17, 28, 0.80)
            );

        border:
            1px solid rgba(148, 163, 184, 0.10);

        border-radius: 20px;
    }

    .waiting-icon {
        font-size: 42px;

        margin-bottom: 15px;

        opacity: 0.8;
    }

    .waiting-title {
        font-size: 18px;

        font-weight: 700;

        color: #cbd5e1;

        margin-bottom: 7px;
    }

    .waiting-description {
        color: #64748b;

        font-size: 13px;

        max-width: 250px;
    }


    /* Footer */

    .footer {
        text-align: center;

        margin-top: 35px;

        padding-top: 20px;

        border-top:
            1px solid rgba(148, 163, 184, 0.08);

        color: #475569;

        font-size: 11px;

        letter-spacing: 1px;
    }


    /* Hide Streamlit Default Elements */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }


    /* Responsive Design */

    @media (max-width: 768px) {

        .brand-name {
            font-size: 28px;
        }

        .brand-tagline {
            margin-left: 0;
        }

        .result-real,
        .result-fake {
            font-size: 34px;
        }

        .glass-card {
            padding: 18px;
        }
    }

</style>
""", unsafe_allow_html=True)


# Header

st.markdown(
    """
    <div class="brand-container">

        <div class="brand-icon">
            🔎
        </div>

        <div class="brand-name">
            FRAMECHECK
        </div>

    </div>

    <div class="brand-tagline">
        AI-powered image authenticity analysis
    </div>
    """,
    unsafe_allow_html=True
)


# Main Layout

left_column, right_column = st.columns(
    [1.15, 0.85],
    gap="large"
)


# Image Upload Section

with left_column:

    st.markdown(
        '<div class="section-label">01 • Image Input</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="glass-card">

            <div class="upload-title">
                Upload an image
            </div>

            <div class="upload-description">
                Upload a facial image to analyze its authenticity.
            </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Drop your image here or browse files",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )

    st.markdown(
        """
        </div>
        """,
        unsafe_allow_html=True
    )


    # Image Preview

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.markdown(
            '<div class="section-label">02 • Image Preview</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True
        )

        st.image(
            image,
            caption="Uploaded image",
            width="stretch"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# Analysis Section

with right_column:

    st.markdown(
        '<div class="section-label">03 • Analysis</div>',
        unsafe_allow_html=True
    )


    # Waiting State

    if uploaded_file is None:

        st.markdown(
            """
            <div class="waiting-card">

                <div class="waiting-icon">
                    ✦
                </div>

                <div class="waiting-title">
                    Awaiting Image
                </div>

                <div class="waiting-description">
                    Upload an image on the left to start the
                    authenticity analysis.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # Prediction

    else:

        # Transform Image

        image_tensor = valid_test_transform(image)

        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(device)


        # Model Prediction

        with torch.no_grad():

            outputs = model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )


        # Get Prediction

        predicted_class = probabilities.argmax(
            dim=1
        ).item()

        confidence = probabilities.max().item()


        # Class Names

        class_names = ["Fake", "Real"]

        prediction = class_names[predicted_class]

        confidence_percentage = confidence * 100


        # Result Styling

        if prediction == "Real":

            result_class = "result-real"

            result_icon = "✓"

        else:

            result_class = "result-fake"

            result_icon = "⚠"


        # Result Card

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Classification
                </div>

                <div class="{result_class}">
                    {result_icon} {prediction.upper()}
                </div>

                <div class="confidence-title">

                    <span>
                        Model Confidence
                    </span>

                    <span class="confidence-value">
                        {confidence_percentage:.2f}%
                    </span>

                </div>

                <div class="confidence-track">

                    <div
                        class="confidence-fill"
                        style="width: {confidence_percentage:.2f}%;">
                    </div>

                </div>

                <div class="details-grid">

                    <div class="detail-item">

                        <div class="detail-title">
                            Model
                        </div>

                        <div class="detail-value">
                            Custom CNN
                        </div>

                    </div>

                    <div class="detail-item">

                        <div class="detail-title">
                            Framework
                        </div>

                        <div class="detail-value">
                            PyTorch
                        </div>

                    </div>

                    <div class="detail-item">

                        <div class="detail-title">
                            Input
                        </div>

                        <div class="detail-value">
                            128 × 128
                        </div>

                    </div>

                    <div class="detail-item">

                        <div class="detail-title">
                            Device
                        </div>

                        <div class="detail-value">
                            CPU
                        </div>

                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# Footer

st.markdown(
    """
    <div class="footer">

        FRAMECHECK &nbsp;•&nbsp;
        CUSTOM CNN &nbsp;•&nbsp;
        PYTORCH &nbsp;•&nbsp;
        IMAGE AUTHENTICITY ANALYSIS

    </div>
    """,
    unsafe_allow_html=True
)
