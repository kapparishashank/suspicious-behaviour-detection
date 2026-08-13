import os
import cv2
import shutil
from ultralytics import YOLO
from tqdm import tqdm

# Paths
SOURCE_DIR = r'c:\Users\sadam\OneDrive\Documents\Field project\processed_data'
DEST_DIR = r'c:\Users\sadam\OneDrive\Documents\Field project\yolo_data'
DEVICE = 'cuda' # Use GPU for fast labeling

def prepare_yolo_dataset(split, max_images=1000):
    print(f"Preparing YOLO data for {split}...")
    model = YOLO('yolov8n.pt') # Using small model for speed
    
    # Target folders
    img_dest = os.path.join(DEST_DIR, split, 'images')
    lbl_dest = os.path.join(DEST_DIR, split, 'labels')
    
    count = 0
    # Process both classes 0 (Normal) and 1 (Anomaly)
    for class_id in ['0', '1']:
        class_src = os.path.join(SOURCE_DIR, split, class_id)
        if not os.path.exists(class_src): continue
            
        files = [f for f in os.listdir(class_src) if f.endswith('.jpg')]
        # Select subset
        files = files[:max_images // 2]
        
        for f in tqdm(files, desc=f"Processing Class {class_id}"):
            src_path = os.path.join(class_src, f)
            
            # Detect
            results = model(src_path, device=DEVICE, verbose=False)
            
            # Filter for humans (Class 0 in COCO)
            yolo_labels = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if int(box.cls) == 0: # 0 is 'person'
                        # YOLO format: class x_center y_center width height (normalized)
                        xywhn = box.xywhn[0].tolist()
                        yolo_labels.append(f"0 {xywhn[0]} {xywhn[1]} {xywhn[2]} {xywhn[3]}")
            
            # Only save if humans detected
            if yolo_labels:
                # Copy image
                shutil.copy(src_path, os.path.join(img_dest, f))
                # Save label
                with open(os.path.join(lbl_dest, f.replace('.jpg', '.txt')), 'w') as lf:
                    lf.write("\n".join(yolo_labels))
                count += 1
                
    print(f"Finished {split}. Total annotated images: {count}")

def create_yaml():
    yaml_content = f"""
path: {DEST_DIR}
train: train/images
val: val/images

names:
  0: human
"""
    with open(os.path.join(r'c:\Users\sadam\OneDrive\Documents\Field project', 'data.yaml'), 'w') as f:
        f.write(yaml_content.strip())
    print("Created data.yaml")

if __name__ == "__main__":
    prepare_yolo_dataset('train', max_images=1000)
    prepare_yolo_dataset('valid', max_images=200) # Use valid as 'val' for YOLO
    create_yaml()
