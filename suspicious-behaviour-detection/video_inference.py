import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import cv2
from PIL import Image
import os
import time

# Configuration
YOLO_MODEL_PATH = r'c:\Users\sadam\OneDrive\Documents\Field project\runs\detect\yolo_runs\human_detection2\weights\best.pt'
CLASSIFIER_PATH = r'c:\Users\sadam\OneDrive\Documents\Field project\best_pretrained_mobilenetv2.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Models
print(f"Initializing video processing on {DEVICE}...")
det_model = YOLO(YOLO_MODEL_PATH)
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

def process_video(input_path, output_path='analyzed_video.mp4', frame_skip=2):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {total_frames} frames. Output will be saved to {output_path}")

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        
        # Performance optimization: Skip frames
        if frame_count % frame_skip != 0:
            out.write(frame)
            continue

        # Convert for internal processing
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Human Detection
        results = det_model(frame, device=DEVICE, verbose=False, conf=0.4)
        
        frame_is_suspicious = False
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # 2. Crop & Classify Behavior
                crop = img_rgb[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0: continue
                
                crop_pil = Image.fromarray(crop)
                input_tensor = preprocess(crop_pil).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    outputs = class_model(input_tensor)
                    _, preds = torch.max(outputs, 1)
                    score = torch.softmax(outputs, dim=1)[0][preds].item()
                    label = "Suspicious" if preds.item() == 1 else "Normal"
                
                # 3. Draw Results
                color = (0, 0, 255) if label == "Suspicious" else (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{label} {score:.1%}", (int(x1), int(y1)-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                if label == "Suspicious":
                    frame_is_suspicious = True

        # Global system alert on frame
        if frame_is_suspicious:
            cv2.putText(frame, "!!! ANOMALY DETECTED !!!", (width//2 - 150, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # Draw red border on frame
            cv2.rectangle(frame, (0,0), (width, height), (0,0,255), 10)

        out.write(frame)
        
        if frame_count % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {frame_count}/{total_frames} frames... ({frame_count/elapsed:.1f} FPS)")

    cap.release()
    out.release()
    print(f"\nProcessing Complete! Total time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    # Example video from original dataset (Walking split)
    video_to_test = r'c:\Users\sadam\OneDrive\Documents\Field project\archive\dataset-video-split\test\Walking001_x264.mp4'
    if os.path.exists(video_to_test):
        process_video(video_to_test, 'analyzed_output.mp4')
    else:
        print(f"Path not found: {video_to_test}. Please provide a valid .mp4 path.")
