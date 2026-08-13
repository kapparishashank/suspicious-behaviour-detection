from ultralytics import YOLO
import os

# Paths
DATA_YAML = r'c:\Users\sadam\OneDrive\Documents\Field project\data.yaml'

def train_custom_yolo():
    print("Starting YOLOv8 Training...")
    # Load a pretrained model
    model = YOLO('yolov8n.pt') 
    
    # Train the model
    # Results will be saved to 'runs/detect/train'
    results = model.train(
        data=DATA_YAML,
        epochs=10,
        imgsz=224, # Smaller images for speed
        batch=16,
        device='cuda',
        project='yolo_runs',
        name='human_detection'
    )
    
    print("\nTraining Complete!")
    print(f"Best model saved at: {os.path.join('yolo_runs', 'human_detection', 'weights', 'best.pt')}")
    
    # Metrics
    metrics = model.val() # Evaluate on validation set
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

if __name__ == "__main__":
    if not os.path.exists(DATA_YAML):
        print("Error: data.yaml not found. Run prepare_yolo_data.py first.")
    else:
        train_custom_yolo()
