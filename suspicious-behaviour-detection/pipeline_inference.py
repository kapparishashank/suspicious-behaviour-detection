import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import cv2
from PIL import Image
import os
import numpy as np

# Configuration
YOLO_MODEL_PATH = r'c:\Users\sadam\OneDrive\Documents\Field project\runs\detect\yolo_runs\human_detection\weights\best.pt'
if not os.path.exists(YOLO_MODEL_PATH): # Fallback for secondary runs
    YOLO_MODEL_PATH = r'c:\Users\sadam\OneDrive\Documents\Field project\runs\detect\yolo_runs\human_detection2\weights\best.pt'

CLASSIFIER_PATH = r'c:\Users\sadam\OneDrive\Documents\Field project\best_pretrained_mobilenetv2.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Models
print("Loading YOLOv8 Detection Model...")
det_model = YOLO(YOLO_MODEL_PATH)

print("Loading MobileNetV2 Classification Model...")
class_model = models.mobilenet_v2(pretrained=False)
class_model.classifier[1] = nn.Linear(class_model.last_channel, 2)
class_model.load_state_dict(torch.load(CLASSIFIER_PATH))
class_model = class_model.to(DEVICE)
class_model.eval()

# 2. Classification Transform
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def run_pipeline(image_path, output_path='result.jpg'):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run YOLO Detection
    results = det_model(image_path, device=DEVICE, verbose=False)
    
    suspicious_count = 0
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            
            # Crop human region
            crop = img_rgb[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0: continue
            
            # Predict using MobileNetV2
            crop_pil = Image.fromarray(crop)
            input_tensor = preprocess(crop_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = class_model(input_tensor)
                _, preds = torch.max(outputs, 1)
                score = torch.softmax(outputs, dim=1)[0][preds].item()
                label = "Suspicious" if preds.item() == 1 else "Normal"
            
            # Visualization
            color = (0, 0, 255) if label == "Suspicious" else (0, 255, 0)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, f"{label} ({score:.2f})", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if label == "Suspicious": suspicious_count += 1

    # Final Alert
    alert_text = "ALERT: Suspicious Activity Detected!" if suspicious_count > 0 else "System Status: Normal Behavior"
    alert_color = (0, 0, 255) if suspicious_count > 0 else (0, 255, 0)
    cv2.putText(img, alert_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, alert_color, 3)

    cv2.imwrite(output_path, img)
    print(f"Result saved to {output_path}")
    print(f"Summary: {alert_text}")

if __name__ == "__main__":
    # Test on a sample image from the test set
    test_img = r'c:\Users\sadam\OneDrive\Documents\Field project\processed_data\test\1'
    if os.path.exists(test_img):
        sample = os.path.join(test_img, os.listdir(test_img)[0])
        run_pipeline(sample, 'pipeline_result.jpg')
    else:
        print("Please ensure processed_data exists.")
