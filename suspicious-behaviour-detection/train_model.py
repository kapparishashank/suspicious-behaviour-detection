import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time

# Configuration
DATA_DIR = r'c:\Users\sadam\OneDrive\Documents\Field project\processed_data_cropped'
BATCH_SIZE = 32
NUM_EPOCHS = 12 # Slightly more epochs for potentially better convergence
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPORT_FILE = 'model_performance_report.txt'

def log_to_file(content):
    with open(REPORT_FILE, 'a') as f:
        f.write(content + '\n')
    print(content)

def train_and_evaluate(model_name, model_arch):
    log_to_file(f"\n{'='*30}")
    log_to_file(f"Model: {model_name}")
    log_to_file(f"{'='*30}")
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                      for x in ['train', 'valid', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
                   for x in ['train', 'valid', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

    model = model_arch.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    best_model_path = f'best_{model_name.replace(" ", "_").lower()}.pth'

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        for phase in ['train', 'valid']:
            if phase == 'train': model.train()
            else: model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_path)
    
    total_time = time.time() - start_time
    log_to_file(f"Training Time: {total_time:.2f}s")
    log_to_file(f"Best Validation Accuracy: {best_acc:.4f}")

    # Test
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    normal_acc = cm[0,0] / np.sum(cm[0,:]) if np.sum(cm[0,:]) > 0 else 0
    anomaly_acc = cm[1,1] / np.sum(cm[1,:]) if np.sum(cm[1,:]) > 0 else 0
    overall_acc = np.mean(np.array(all_preds) == np.array(all_labels))

    log_to_file("\n--- Test Set Results ---")
    log_to_file(f"Overall Accuracy (Independent): {overall_acc:.4f}")
    log_to_file(f"Normal Accuracy (Object Dependent): {normal_acc:.4f}")
    log_to_file(f"Anomaly Accuracy (Object Dependent): {anomaly_acc:.4f}")
    log_to_file("\nConfusion Matrix:")
    log_to_file(str(cm))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'cm_{model_name.replace(" ", "_").lower()}.png')
    plt.close()

if __name__ == "__main__":
    # We don't remove the REPORT_FILE here, just append
    log_to_file(f"\nAdding new model to evaluation on {DEVICE}")

    # MobileNetV2 (Matches inference pipeline)
    mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Binary classification head
    mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 2)
    
    train_and_evaluate("Pretrained MobileNetV2", mobilenet)

    print("\nTraining complete. Check model_performance_report.txt for the updated comparison.")
