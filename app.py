import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
# import json # Not needed, removed
# from pathlib import Path # Not needed, removed
import numpy as np

# -------------------------
# Custom CSS (Enhanced for a professional look)
# -------------------------
st.markdown("""
<style>
/* 1. Global Background & Font */
.stApp {
    background: #f0f2f6; /* Very light grey/blue for a soft background */
    font-family: 'Inter', 'Segoe UI', sans-serif; /* Modern, clean font */
}

/* 2. Header/Title Styling */
h1 {
    color: #0d47a1; /* Deep blue for a strong brand color */
    text-align: center;
    padding-top: 15px;
    padding-bottom: 5px;
    border-bottom: 2px solid #e0e0e0; /* Subtle line under the title */
    margin-bottom: 30px;
}

/* 3. Card Styling (More defined) */
.st-emotion-cache-1c9v62e { /* Target the main block container */
    background-color: white;
    padding: 30px;
    border-radius: 12px; /* Smoother corners */
    box-shadow: 0 4px 12px rgba(0,0,0,0.05); /* Softer shadow */
    margin-top: 25px;
    border: 1px solid #e0e0e0; /* Subtle border */
}

/* 4. Custom Prediction Card (Primary result focus) */
.prediction-result-card {
    background-color: #e3f2fd; /* Lightest blue background */
    padding: 35px 25px;
    border-radius: 12px;
    box-shadow: 0 6px 15px rgba(13, 71, 161, 0.1); /* Branded shadow */
    margin-top: 20px;
    border-left: 5px solid #0d47a1; /* Strong left border for emphasis */
}

.pred-label {
    font-size: 32px;
    font-weight: 800; /* Extra bold */
    color: #0d47a1;
    margin-bottom: 5px;
    text-transform: uppercase;
}

.confidence {
    font-size: 22px;
    color: #388e3c; /* Green for success/confidence */
    font-weight: 600;
}

/* 5. Streamlit Button Styling */
.stButton>button {
    background-color: #0d47a1;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
    transition: all 0.2s ease-in-out;
}

.stButton>button:hover {
    background-color: #1565c0; /* Slightly lighter blue on hover */
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* 6. Sidebar (More professional dark theme) */
[data-testid="stSidebar"] {
    background: #263238; /* Darker, less aggressive navy */
    color: white;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] p {
    color: #e0e0e0;
}
</style>
""", unsafe_allow_html=True)


# -------------------------
# CONFIG
# -------------------------
# Path needs to be correct for the app to run. Assuming 'best_model.pth' is in the same directory.
MODEL_PATH = "best_model.pth"
IMG_SIZE = 224
TOPK = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Build model (same as training)
# -------------------------
def build_model(model_name, num_classes):
    # Model structure unchanged
    if model_name.lower() == "resnet50":
        # Using weights=None to avoid warning when loading checkpoint
        model = models.resnet50(weights=None) 
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    else:
        # Added st.error for visibility in the app
        st.error("Only ResNet50 supported in this app") 
        raise ValueError("Only ResNet50 supported in this app")
    return model

# -------------------------
# Load model & metadata
# -------------------------
@st.cache_resource
def load_model():
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

        model_name = checkpoint.get("model_name", "resnet50")
        class_to_idx = checkpoint.get("class_to_idx")
        if not class_to_idx:
             st.error("Error: 'class_to_idx' not found in model checkpoint.")
             raise KeyError("'class_to_idx' not found")
             
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(idx_to_class)

        model = build_model(model_name, num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        model.eval()

        return model, idx_to_class, model_name
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure the file is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during model loading: {e}")
        st.stop()


# Load model globally (or use st.session_state if hot-reloading causes issues)
try:
    model, idx_to_class, model_name = load_model()
except Exception:
    # Stop execution if loading fails to prevent subsequent errors
    st.stop() 


# -------------------------
# Prediction function
# -------------------------
def predict_image(image, topk=3):
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    topk_idx = probs.argsort()[-topk:][::-1]
    # Ensure correct case for display
    results = [(idx_to_class[i].capitalize(), probs[i]) for i in topk_idx] 
    return results

# -------------------------
# Streamlit UI (Refined Layout)
# -------------------------
st.set_page_config(page_title="Smart Waste AI Classifier", page_icon="♻️", layout="centered")

# --- Main Header ---
st.title("♻️ Smart Waste AI Classifier")

# --- Introduction & Description ---
st.info(
    "Upload an image of waste below. Our **ResNet50-based AI** model will instantly classify it into one of six categories to help you with proper sorting and recycling.",
    icon="💡"
)

# --- Main Content Layout (Image Upload & Prediction) ---
# Use columns to balance the uploaded image and the prediction results
upload_col, info_col = st.columns([1, 1])

uploaded_file = upload_col.file_uploader("📤 Choose an Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display image in the left column
    image = Image.open(uploaded_file).convert("RGB")
    upload_col.image(image, caption="Uploaded Item", width=True)

    # Prediction logic in the right column
    if info_col.button("🚀 Analyze Waste"):
        with st.spinner("🔍 Running AI classification..."):
            predictions = predict_image(image, TOPK)

        top_label, top_prob = predictions[0]

        # --- Primary Prediction Card ---
        info_col.markdown("### ✨ Top Result")
        info_col.markdown(
            f"""
            <div class="prediction-result-card">
                <div class="pred-label">{top_label}</div>
                <div class="confidence">Confidence: {top_prob*100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # --- Detailed Predictions (Collapsed Section) ---
        with info_col.expander("Show Detailed Confidence Scores"):
            for label, prob in predictions:
                st.write(f"**{label}**")
                # Use st.progress for a nicer visual display of probability
                st.progress(float(prob)) 
                st.write(f"{prob*100:.2f}%")




# --- Separator and Footer Info ---
st.markdown("---")

# -------------------------
# Sidebar info (Unchanged, but looks better with new CSS)
# -------------------------
st.sidebar.title("ℹ️ Technical Details")
st.sidebar.markdown("---")
st.sidebar.write(f"🧠 **Model:** {model_name.upper()} (Deep Learning)")
st.sidebar.write(f"⚙️ **Processing:** {str(DEVICE).upper()}")
st.sidebar.write(f"📏 **Input Size:** {IMG_SIZE}x{IMG_SIZE}px")
st.sidebar.markdown("---")

# Instruction block in the sidebar
st.sidebar.header("Categories")
st.sidebar.markdown(
    """
    The model classifies waste into 6 categories:
    * **Cardboard**
    * **Glass**
    * **Metal**
    * **Paper**
    * **Plastic**
    * **Trash** (Non-recyclable)
    """
)