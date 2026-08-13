import cv2
import os
import random
import torch
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm

# Paths
base_dir = r'c:\Users\sadam\OneDrive\Documents\Field project\archive\dataset-video-split'
output_dir = r'c:\Users\sadam\OneDrive\Documents\Field project\processed_data_cropped'
# YOLO_MODEL_PATH = r'c:\Users\sadam\OneDrive\Documents\Field project\runs\detect\yolo_runs\human_detection2\weights\best.pt'
YOLO_MODEL_PATH = 'yolov8n.pt' # Fallback to base model if unsure about path availability across environments
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Class mapping
anomalous_prefixes = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
normal_prefixes = ['Clapping', 'Meet_and_Split', 'Normal_Videos', 'Sitting', 'Standing_Still', 'Walking', 'Walking_While_Reading_Book', 'Walking_While_Using_Phone']

# Configuration
FRAMES_PER_VIDEO = 15 # More frames to ensure we catch humans
RESIZE_DIM = (224, 224)

# Load YOLO
print(f"Loading YOLO for cropping on {DEVICE}...")
yolo_model = YOLO(YOLO_MODEL_PATH)

def get_binary_label(filename):
    for prefix in anomalous_prefixes:
        if filename.startswith(prefix):
            return "Anomaly"
    for prefix in normal_prefixes:
        if filename.startswith(prefix):
            return "Normal"
    return None

def extract_cropped_humans(video_path, output_path, n_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return 0
    
    step = max(total_frames // n_frames, 1)
    
    extracted = 0
    for i in range(n_frames):
        frame_idx = i * step
        if frame_idx >= total_frames:
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect Humans
        results = yolo_model(frame, device=DEVICE, verbose=False, conf=0.4)
        
        for result in results:
            boxes = result.boxes
            for idx, box in enumerate(boxes):
                if int(box.cls) != 0: continue # Only humans
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                
                if crop.size == 0: continue
                
                # Resize and Save
                crop = cv2.resize(crop, RESIZE_DIM)
                img_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_f{i}_p{idx}.jpg"
                cv2.imwrite(os.path.join(output_path, img_name), crop)
                extracted += 1
                
    cap.release()
    return extracted

def process_split(split_name):
    print(f"\n--- Processing {split_name} split ---")
    split_src = os.path.join(base_dir, split_name)
    
    if not os.path.exists(split_src):
        print(f"Directory {split_src} not found.")
        return
        
    split_dst_anomaly = os.path.join(output_dir, split_name, '1') # 1 = Anomaly
    split_dst_normal = os.path.join(output_dir, split_name, '0')  # 0 = Normal
    
    os.makedirs(split_dst_anomaly, exist_ok=True)
    os.makedirs(split_dst_normal, exist_ok=True)
    
    videos = [f for f in os.listdir(split_src) if f.endswith('.mp4')]
    results = {'Anomaly': 0, 'Normal': 0}
    
    for video in tqdm(videos, desc=f"Videos in {split_name}"):
        label = get_binary_label(video)
        if not label: continue
            
        src_path = os.path.join(split_src, video)
        dst_path = split_dst_anomaly if label == 'Anomaly' else split_dst_normal
        
        extracted = extract_cropped_humans(src_path, dst_path, FRAMES_PER_VIDEO)
        results[label] += extracted
        
    print(f"Extracted for {split_name}: {results}")

def balance_split(split_name):
    print(f"Balancing {split_name} split...")
    anomaly_dir = os.path.join(output_dir, split_name, '1')
    normal_dir = os.path.join(output_dir, split_name, '0')
    
    if not os.path.exists(anomaly_dir) or not os.path.exists(normal_dir): return
    
    anomaly_files = os.listdir(anomaly_dir)
    normal_files = os.listdir(normal_dir)
    
    a_count = len(anomaly_files)
    n_count = len(normal_files)
    
    if a_count == 0 or n_count == 0: return
        
    target = min(a_count, n_count)
    
    if a_count > target:
        to_delete = random.sample(anomaly_files, a_count - target)
        for f in to_delete: os.remove(os.path.join(anomaly_dir, f))
        
    if n_count > target:
        to_delete = random.sample(normal_files, n_count - target)
        for f in to_delete: os.remove(os.path.join(normal_dir, f))
        
    print(f"Final counts for {split_name}: {target} images per class.")

if __name__ == "__main__":
    for split in ['train', 'valid', 'test']:
        process_split(split)
        balance_split(split)
    print("\nCropped Dataset Preparation Complete!")
