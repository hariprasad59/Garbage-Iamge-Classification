import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

# -------------------------
# CONFIG
# -------------------------
REPO_ID = "HariPrasad599/garbage-image-classifier"
MODEL_FILENAME = "best_model.pth"
IMG_SIZE = 224
TOPK = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Smart Waste AI Classifier", page_icon="♻️", layout="centered")

# -------------------------
# Download model from Hugging Face
# -------------------------
@st.cache_resource
def download_model():
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME
    )

MODEL_PATH = download_model()

# -------------------------
# Image preprocessing
# -------------------------
preprocess = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# -------------------------
# Build model
# -------------------------
def build_model(num_classes):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    return model

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, idx_to_class

model, idx_to_class = load_model()

# -------------------------
# Prediction
# -------------------------
def predict_image(image):
    img = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(img), dim=1)[0].cpu().numpy()
    topk_idx = probs.argsort()[-TOPK:][::-1]
    return [(idx_to_class[i].capitalize(), probs[i]) for i in topk_idx]

# -------------------------
# UI (ONLY ONCE)
# -------------------------
st.title("♻️ Smart Waste AI Classifier")
st.info("Upload a waste image and the AI will classify it.")

col1, col2 = st.columns(2)

file = col1.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file).convert("RGB")
    col1.image(image, caption="Uploaded Image", use_container_width=True)

    if col2.button("🚀 Predict"):
        preds = predict_image(image)
        label, prob = preds[0]

        col2.success(f"**{label}** — {prob*100:.2f}% confidence")

        with col2.expander("🔍 Top predictions"):
            for l, p in preds:
                st.write(f"{l}: {p*100:.2f}%")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("ℹ️ Model Info")
st.sidebar.write("**Architecture:** ResNet50")
st.sidebar.write("**Input Size:** 224 × 224")
st.sidebar.write("**Classes:**")
st.sidebar.markdown("""
- Cardboard  
- Glass  
- Metal  
- Paper  
- Plastic  
- Trash  
""")
