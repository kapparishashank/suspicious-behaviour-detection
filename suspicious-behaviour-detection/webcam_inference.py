import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import cv2
from PIL import Image
import time

# Configuration
YOLO_MODEL_PATH = r'c:\Users\sadam\OneDrive\Documents\Field project\runs\detect\yolo_runs\human_detection2\weights\best.pt'
CLASSIFIER_PATH = r'c:\Users\sadam\OneDrive\Documents\Field project\best_pretrained_mobilenetv2.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Models
print(f"Starting Real-time Webcam AI on {DEVICE}...")
det_model = YOLO(YOLO_MODEL_PATH)
class_model = models.mobilenet_v2(pretrained=False)
class_model.classifier[1] = nn.Linear(class_model.last_channel, 2)
class_model.load_state_dict(torch.load(CLASSIFIER_PATH))
class_model = class_model.to(DEVICE)
class_model.eval()

# 2. Classification Transform & Sensitivity Settings
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

CONF_THRESHOLD = 0.90  # Increased for higher precision
SMOOTHING_FRAMES = 8   # Require more frames of consistency
abnormal_history = []  # Buffer to store recent detections
ALARM_COUNT_THRESHOLD = 2 # Only trigger alert if at least 2 people are abnormal (reduce noise in crowds)

def run_webcam():
    global abnormal_history
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("--- Sensitivity Adjusted: Confidence (>85%) + Smoothing Enabled ---")
    print("--- Press 'q' to Exit ---")
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = det_model(frame, device=DEVICE, verbose=False, conf=0.5)
        
        # Count how many abnormal people in THIS frame
        frame_abnormal_count = 0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                crop = img_rgb[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0: continue
                
                crop_pil = Image.fromarray(crop)
                input_tensor = preprocess(crop_pil).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    outputs = class_model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                    # Class 1 is Abnormal
                    abnormal_prob = probabilities[1].item()
                    
                    # Logically determine if it meets our strict criteria
                    if abnormal_prob > CONF_THRESHOLD:
                        frame_abnormal_count += 1
                        label = "Abnormal"
                        score = abnormal_prob
                    else:
                        label = "Normal"
                        score = probabilities[0].item()
                
                color = (0, 0, 255) if label == "Abnormal" else (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{label} {score:.1%}", (int(x1), int(y1)-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Apply Smoothing Logic: Frame is abnormal if ANY person is very suspicious
        current_frame_is_flagged = (frame_abnormal_count > 0)
        abnormal_history.append(current_frame_is_flagged)
        
        if len(abnormal_history) > SMOOTHING_FRAMES:
            abnormal_history.pop(0)
        
        # Require 60% of the window to be abnormal to trigger the "BIG RED ALERT"
        alert_condition = sum(abnormal_history) >= (len(abnormal_history) * 0.6)
        
        if alert_condition:
            cv2.putText(frame, "!!! SECURITY ALERT: ABNORMAL !!!", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # Flash red border
            cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 10)

        # Performance Overlay
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-100, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # Show frame
        cv2.imshow('AI Security Pipeline (Webcam)', frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        run_webcam()
    except Exception as e:
        print(f"Error during execution: {e}")
