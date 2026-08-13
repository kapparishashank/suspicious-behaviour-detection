import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import cv2
from PIL import Image
import os
import tempfile
import time
import pandas as pd

# --- UI PRESETS ---
st.set_page_config(
    page_title="AI Surveillance Insight",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stAlert {
        border-radius: 10px;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #ffffff;
    }
    .css-1offfwp {
        background-color: #161b22 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURATION (Using relative paths for GitHub) ---
YOLO_MODEL_PATH = './models/yolo_runs/human_detection/weights/best.pt'
CLASSIFIER_PATH = './models/best_mobilenetv2.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    # Check if model files exist before loading
    if not os.path.exists(YOLO_MODEL_PATH) or not os.path.exists(CLASSIFIER_PATH):
        st.error("❌ Model weights not found! Please ensure models are in the `./models/` directory.")
        return None, None
        
    try:
        det_model = YOLO(YOLO_MODEL_PATH)
        # Fixed PyTorch deprecation warning (pretrained=False -> weights=None)
        class_model = models.mobilenet_v2(weights=None)
        class_model.classifier[1] = nn.Linear(class_model.last_channel, 2)
        class_model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
        class_model = class_model.to(DEVICE).eval()
        return det_model, class_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main():
    # Sidebar Branding
    with st.sidebar:
        st.image("https://img.icons8.com/isometric-folders/100/security-configuration.png", width=80)
        st.title("Settings")
        st.markdown("---")
        conf_threshold = st.slider("Anomaly Sensitivity", 0.5, 1.0, 0.85, step=0.01)
        frame_skip = st.number_input("Frame Skip (1 = every frame)", 1, 10, 2)
        st.markdown("---")
        st.info("High sensitivity might result in more false positives but captures subtle anomalies.")

    # Header Section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🛡️ AI Surveillance & Efficiency Insight")
        st.markdown("Advanced behavior analysis using YOLOv8 + MobileNetV2")
    with col2:
        st.empty() # Placeholder for logo or time

    det_model, class_model = load_models()
    if not det_model: return

    # File Upload Area
    uploaded_file = st.file_uploader("Upload Video for Behavior Verification", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file:
        # Create a temporary file to save the uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
        tfile.write(uploaded_file.read())
        temp_file_path = tfile.name
        
        cap = cv2.VideoCapture(temp_file_path)
        
        if not cap.isOpened():
            st.error("❌ Could not open video file. It might be corrupted.")
            os.remove(temp_file_path) # Clean up
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Dashboard Layout
        m1, m2, m3 = st.columns(3)
        fps_metric = m1.metric("Processing Speed", "0 FPS")
        person_metric = m2.metric("Active Detection", "0 People")
        status_metric = m3.metric("Security Status", "Neutral")
        
        st.markdown("### Live Analysis Stream")
        video_placeholder = st.empty()
        
        progress_bar = st.progress(0)
        
        # Stats accumulation
        raw_frame_count = 0
        suspicion_trend = []
        
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                raw_frame_count += 1
                
                # Update progress bar even for skipped frames
                progress_bar.progress(min(raw_frame_count / total_frames, 1.0))

                # Skip frames to speed up processing
                if raw_frame_count % frame_skip != 0:
                    continue

                # Processing Timer
                proc_start = time.time()
                
                # Inference
                results = det_model(frame, device=DEVICE, verbose=False, 