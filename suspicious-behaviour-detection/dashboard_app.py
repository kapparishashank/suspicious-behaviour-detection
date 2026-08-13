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

# --- CONFIGURATION ---
YOLO_MODEL_PATH = r'c:\Users\sadam\OneDrive\Documents\Field project\runs\detect\yolo_runs\human_detection2\weights\best.pt'
CLASSIFIER_PATH = r'c:\Users\sadam\OneDrive\Documents\Field project\best_pretrained_mobilenetv2.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    try:
        det_model = YOLO(YOLO_MODEL_PATH)
        class_model = models.mobilenet_v2(pretrained=False)
        class_model.classifier[1] = nn.Linear(class_model.last_channel, 2)
        class_model.load_state_dict(torch.load(CLASSIFIER_PATH))
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
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
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
        frames_processed = 0
        all_detections = []
        suspicion_trend = []
        
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frames_processed += 1
            if frames_processed % frame_skip != 0:
                continue

            # Update progress
            progress_bar.progress(min(frames_processed / total_frames, 1.0))
            
            # Processing Timer
            proc_start = time.time()
            
            # Inference
            results = det_model(frame, device=DEVICE, verbose=False, conf=0.5)
            
            current_people_count = 0
            is_abnormal = False
            max_suspicion = 0.0
            
            for result in results:
                for box in result.boxes:
                    current_people_count += 1
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    
                    if crop.size == 0: continue
                    
                    # Classifier Preprocessing
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    input_tensor = preprocess(Image.fromarray(crop_rgb)).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        probs = torch.softmax(class_model(input_tensor), dim=1)[0]
                        abnormal_prob = probs[1].item()
                        max_suspicion = max(max_suspicion, abnormal_prob)
                        
                        if abnormal_prob > conf_threshold:
                            is_abnormal = True
                            label, color = "⚠️ SUSPICIOUS", (0, 0, 255)
                        else:
                            label, color = "Normal", (0, 255, 0)
                    
                    # Visualization
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{label} {abnormal_prob:.1%}", (int(x1), int(y1)-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Global Frame Status
            if is_abnormal:
                cv2.putText(frame, "ANOMALY DETECTED", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                status_metric.metric("Security Status", "🚨 ALERT", delta="-Suspicious Activity")
            else:
                status_metric.metric("Security Status", "✅ SECURE", delta="Normal")

            # Metrics Calculation
            proc_time = time.time() - proc_start
            current_fps = 1.0 / proc_time if proc_time > 0 else 0
            
            fps_metric.metric("Processing Speed", f"{current_fps:.1f} FPS")
            person_metric.metric("Active Detection", f"{current_people_count} People")
            
            # Update Display
            video_placeholder.image(frame, channels="BGR", use_container_width=True)
            
            # Track trends
            suspicion_trend.append(max_suspicion)

        cap.release()
        st.success(f"Verification complete! Total time: {time.time() - start_time:.1f}s")
        
        # Post-Analysis Stats
        st.markdown("---")
        st.markdown("### Efficiency & Behavior Trends")
        chart_data = pd.DataFrame({"Max Suspicion Level": suspicion_trend})
        st.line_chart(chart_data)
        
        st.info("The trend line shows the highest suspicion score recorded in each frame. Spikes above your threshold represent anomaly detections.")

if __name__ == "__main__":
    main()
