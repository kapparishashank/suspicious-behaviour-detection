import cv2
import os
import random
import shutil

# Paths
base_dir = r'c:\Users\sadam\OneDrive\Documents\Field project\archive\dataset-video-split'
output_dir = r'c:\Users\sadam\OneDrive\Documents\Field project\processed_data'

# Class mapping
anomalous_prefixes = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
normal_prefixes = ['Clapping', 'Meet_and_Split', 'Normal_Videos', 'Sitting', 'Standing_Still', 'Walking', 'Walking_While_Reading_Book', 'Walking_While_Using_Phone']

# Configuration
FRAMES_PER_VIDEO = 10
RESIZE_DIM = (224, 224)

def get_binary_label(filename):
    for prefix in anomalous_prefixes:
        if filename.startswith(prefix):
            return "Anomaly"
    for prefix in normal_prefixes:
        if filename.startswith(prefix):
            return "Normal"
    return None

def extract_frames(video_path, output_path, n_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return 0
    
    # Calculate step to get n_frames evenly spaced
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
            
        # Resize and Save
        frame = cv2.resize(frame, RESIZE_DIM)
        img_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_f{i}.jpg"
        cv2.imwrite(os.path.join(output_path, img_name), frame)
        extracted += 1
        
    cap.release()
    return extracted

def process_split(split_name):
    print(f"Processing {split_name} split...")
    split_src = os.path.join(base_dir, split_name)
    
    # Create output subdirs
    split_dst_anomaly = os.path.join(output_dir, split_name, '1') # 1 = Anomaly
    split_dst_normal = os.path.join(output_dir, split_name, '0')  # 0 = Normal
    
    os.makedirs(split_dst_anomaly, exist_ok=True)
    os.makedirs(split_dst_normal, exist_ok=True)
    
    results = {'Anomaly': 0, 'Normal': 0}
    
    videos = [f for f in os.listdir(split_src) if f.endswith('.mp4')]
    
    for video in videos:
        label = get_binary_label(video)
        if not label:
            print(f"Warning: Unknown label for {video}")
            continue
            
        src_path = os.path.join(split_src, video)
        dst_path = split_dst_anomaly if label == 'Anomaly' else split_dst_normal
        
        extracted = extract_frames(src_path, dst_path, FRAMES_PER_VIDEO)
        results[label] += extracted
        
    print(f"Extracted for {split_name}: {results}")
    return results

def balance_split(split_name):
    print(f"Balancing {split_name} split...")
    anomaly_dir = os.path.join(output_dir, split_name, '1')
    normal_dir = os.path.join(output_dir, split_name, '0')
    
    anomaly_files = os.listdir(anomaly_dir)
    normal_files = os.listdir(normal_dir)
    
    a_count = len(anomaly_files)
    n_count = len(normal_files)
    
    if a_count == 0 or n_count == 0:
        return
        
    target = min(a_count, n_count)
    
    if a_count > target:
        to_delete = random.sample(anomaly_files, a_count - target)
        for f in to_delete:
            os.remove(os.path.join(anomaly_dir, f))
        print(f"Deleted {len(to_delete)} anomaly frames.")
        
    if n_count > target:
        to_delete = random.sample(normal_files, n_count - target)
        for f in to_delete:
            os.remove(os.path.join(normal_dir, f))
        print(f"Deleted {len(to_delete)} normal frames.")
        
    print(f"Final counts for {split_name}: {target} Exception per class.")

# Main Execution
if __name__ == "__main__":
    if os.path.exists(output_dir):
        # Optional: shutil.rmtree(output_dir) if you want a clean start
        pass
        
    for split in ['train', 'valid', 'test']:
        process_split(split)
        balance_split(split)
        
    print("Process Complete!")
