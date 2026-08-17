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

    .stApp {
        background-color: #0b0f14;
        color: #e8edf2;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 0px;
    }

    .subtitle {
        color: #8f9aa6;
        font-size: 17px;
        margin-top: 4px;
        margin-bottom: 35px;
    }

    .section-label {
        color: #7f8b97;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .info-card {
        background-color: #11171e;
        border: 1px solid #26303a;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 18px;
    }

    .status-title {
        color: #8f9aa6;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .status-text {
        font-size: 24px;
        font-weight: 600;
    }

    .model-info {
        color: #8f9aa6;
        font-size: 13px;
        padding-top: 15px;
        border-top: 1px solid #26303a;
    }

    div[data-testid="stFileUploader"] {
        background-color: #11171e;
        border: 1px dashed #3a4652;
        border-radius: 12px;
        padding: 12px;
    }

</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="main-title">FRAMECHECK</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Image Authenticity Analysis</div>',
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

    with right_column:

        st.markdown(
            '<div class="section-label">Analysis</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="info-card">
                <div class="status-title">CLASSIFICATION</div>
                <div class="status-text">{prediction.upper()}</div>
                <br>
                <div class="model-info">
                    Confidence • {confidence * 100:.2f}%<br>
                    Model • CNN<br>
                    Input Resolution • 128 × 128
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
