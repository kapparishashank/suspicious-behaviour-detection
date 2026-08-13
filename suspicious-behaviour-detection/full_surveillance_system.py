import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

# ==========================================
# 1. CONFIGURATION & MODEL PATHS
# ==========================================
YOLO_MODEL_PATH = r'c:\Users\sadam\OneDrive\Documents\Field project\runs\detect\yolo_runs\human_detection2\weights\best.pt'
CLASSIFIER_PATH = r'c:\Users\sadam\OneDrive\Documents\Field project\best_pretrained_mobilenetv2.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. MODEL DEFINITIONS & LOADING
# ==========================================

def load_surveillance_models():
    """
    Loads both YOLOv8 (Detection) and MobileNetV2 (Classification) models.
    """
    print(f"[System] Loading models on {DEVICE}...")
    
    # Load YOLOv8 for Human Detection
    try:
        det_model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"[Error] Failed to load YOLO: {e}")
        return None, None

    # Load MobileNetV2 for Behavior Classification
    try:
        class_model = models.mobilenet_v2(weights=None)
        # Custom binary head for 'Normal' and 'Suspicious'
        class_model.classifier[1] = nn.Linear(class_model.last_channel, 2)
        class_model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
        class_model = class_model.to(DEVICE).eval()
    except Exception as e:
        print(f"[Error] Failed to load Classifier: {e}")
        return det_model, None

    print("[System] Models loaded successfully.")
    return det_model, class_model

# ==========================================
# 3. PREPROCESSING LOGIC
# ==========================================
# Standard MobileNetV2 normalization
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 4. CORE INFERENCE PIPELINE
# ==========================================

def run_inference(frame, det_model, class_model, conf_threshold=0.85):
    """
    Detects humans and classifies behavior in a single frame.
    """
    # 1. Detection Stage
    results = det_model(frame, device=DEVICE, verbose=False, conf=0.5)
    
    detections = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # 2. Crop Stage
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            # 3. Classification Stage
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(Image.fromarray(crop_rgb)).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = class_model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                suspicion_score = probs[1].item()
                
            # Determine label based on threshold
            is_suspicious = suspicion_score > conf_threshold
            label = "SUSPICIOUS" if is_suspicious else "Normal"
            color = (0, 0, 255) if is_suspicious else (0, 255, 0)
            
            detections.append({
                "box": (x1, y1, x2, y2),
                "label": label,
                "score": suspicion_score,
                "color": color
            })
            
    return detections

# ==========================================
# 5. MAIN EXECUTION (WEBCAM EXAMPLE)
# ==========================================

def main():
    det_model, class_model = load_surveillance_models()
    if not det_model or not class_model:
        return

    # Open Camera
    cap = cv2.VideoCapture(0)
    print("[System] Starting video stream. Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Run the AI Pipeline
        detections = run_inference(frame, det_model, class_model)

        # Visualize Results
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), det["color"], 2)
            cv2.putText(frame, f"{det['label']} ({det['score']:.2%})", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, det["color"], 2)

        cv2.imshow("AI Surveillance Real-Time", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
