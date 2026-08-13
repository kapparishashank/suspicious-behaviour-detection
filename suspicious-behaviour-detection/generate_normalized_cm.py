import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Configuration
DATA_DIR = r'c:\Users\sadam\OneDrive\Documents\Field project\processed_data'
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MobileNetV2 was the best performer on test (89.04%)
MODEL_PATH = 'best_pretrained_mobilenetv2.pth'

def generate_normalized_cm():
    print(f"Loading best model: {MODEL_PATH}")
    
    # Load Model
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(DEVICE)
    model.eval()

    # Data Loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Raw counts
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalization (row-wise: divide by true totals)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plotting
    plt.figure(figsize=(10, 8))
    # We use 'fmt=".2f"' for percentages
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title('Normalized Confusion Matrix: Pretrained MobileNetV2\n(Best Accuracy: 89.04%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    output_img = 'normalized_confusion_matrix.png'
    plt.savefig(output_img)
    print(f"Normalized confusion matrix saved as {output_img}")
    plt.show()

if __name__ == "__main__":
    generate_normalized_cm()
