# train_model.py - Improved version for 90%+ accuracy

import os
import cv2
import numpy as np
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle
import random
from torchvision import transforms
import matplotlib.pyplot as plt

CHARS = string.ascii_uppercase + string.digits
char_to_idx = {ch: i for i, ch in enumerate(CHARS)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}

# Save character mappings for later use
with open('char_mappings.pkl', 'wb') as f:
    pickle.dump({'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}, f)

def preprocess(img):
    """Improved preprocessing with better normalization"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Apply adaptive thresholding for better contrast
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    resized = cv2.resize(gray, (32, 32))  # Increased size for better detail
    norm = resized.astype(np.float32) / 255.0
    return norm.reshape(1, 32, 32)

def split_captcha(img):
    """Improved character splitting with better boundaries"""
    h, w = img.shape[:2]
    char_width = w // 5
    # Add small overlap to avoid cutting characters
    overlap = char_width // 10
    return [img[:, max(0, i * char_width - overlap):min(w, (i + 1) * char_width + overlap)] 
            for i in range(5)]

def augment_image(img):
    """Data augmentation to increase dataset size"""
    augmented = []
    
    # Original image
    augmented.append(img)
    
    # Add noise
    noise = np.random.normal(0, 0.05, img.shape).astype(np.float32)
    noisy_img = np.clip(img + noise, 0, 1)
    augmented.append(noisy_img)
    
    # Slight rotation
    angle = random.uniform(-5, 5)
    h, w = img.shape[1:]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img[0], M, (w, h))
    augmented.append(rotated.reshape(1, 32, 32))
    
    # Slight scaling
    scale = random.uniform(0.9, 1.1)
    scaled = cv2.resize(img[0], None, fx=scale, fy=scale)
    # Pad or crop to maintain size
    h, w = scaled.shape
    if h < 32 or w < 32:
        padded = np.zeros((32, 32), dtype=np.float32)
        padded[:h, :w] = scaled
        augmented.append(padded.reshape(1, 32, 32))
    else:
        cropped = scaled[:32, :32]
        augmented.append(cropped.reshape(1, 32, 32))
    
    # Brightness adjustment
    bright_img = np.clip(img * random.uniform(0.8, 1.2), 0, 1)
    augmented.append(bright_img)
    
    # Contrast adjustment
    contrast_img = np.clip((img - 0.5) * random.uniform(0.8, 1.2) + 0.5, 0, 1)
    augmented.append(contrast_img)
    
    return augmented

class CaptchaDataset(Dataset):
    def __init__(self, X, y, augment=True):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augment and random.random() < 0.5:
            # Apply random augmentation
            x = x.numpy()
            x = augment_image(x)[random.randint(0, 4)]  # Pick one augmentation
            x = torch.FloatTensor(x)
        
        return x, y

def load_data(input_dir, output_dir):
    """Load and augment data"""
    X, y = [], []
    processed_count = 0
    error_count = 0
    
    for fname in os.listdir(input_dir):
        if not fname.endswith('.jpg'):
            continue
        img_path = os.path.join(input_dir, fname)
        label_fname = fname.replace('input', 'output').replace('.jpg', '.txt')
        label_path = os.path.join(output_dir, label_fname)
        
        if not os.path.exists(label_path):
            continue
        
        try:
            with open(label_path, 'r') as f:
                label = f.read().strip()
            
            if len(label) != 5:
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            char_imgs = split_captcha(img)
            
            for i, char_img in enumerate(char_imgs):
                if i < len(label):
                    processed_img = preprocess(char_img)
                    X.append(processed_img)
                    y.append(char_to_idx[label[i]])
            
            processed_count += 1
            
        except Exception as e:
            error_count += 1
            continue
    
    print(f"Processed {processed_count} files, {error_count} errors, {len(X)} character samples.")
    
    # Augment the dataset
    print("Augmenting dataset...")
    X_augmented, y_augmented = [], []
    
    for i in range(len(X)):
        # Original sample
        X_augmented.append(X[i])
        y_augmented.append(y[i])
        
        # Augmented samples (5x more data)
        augmented_samples = augment_image(X[i])
        for aug_sample in augmented_samples[1:]:  # Skip original
            X_augmented.append(aug_sample)
            y_augmented.append(y[i])
    
    print(f"After augmentation: {len(X_augmented)} character samples")
    return np.array(X_augmented), np.array(y_augmented)

class ImprovedCaptchaModel(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCaptchaModel, self).__init__()
        
        # Deeper CNN architecture
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Calculate the size after convolutions
        self._to_linear = 128 * 4 * 4  # 32x32 -> 16x16 -> 8x8 -> 4x4
        
        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=50, device='cpu'):
    """Improved training with learning rate scheduling and early stopping"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_acc = 0
    patience_counter = 0
    patience = 10
    
    train_losses = []
    val_accuracies = []
    
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(running_loss / len(train_loader))
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "best_captcha_model.pth")
            print(f'  New best validation accuracy: {best_val_acc:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  Early stopping after {epoch+1} epochs')
                break
        
        model.train()
    
    # Load best model
    model.load_state_dict(torch.load("best_captcha_model.pth"))
    print(f'Best validation accuracy achieved: {best_val_acc:.2f}%')
    
    return train_losses, val_accuracies

def evaluate_full_captcha_accuracy(model, X, y, device='cpu'):
    """Evaluate full captcha accuracy (all 5 characters correct)"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(X), 5):
            if i + 5 <= len(X):
                x_captcha = torch.FloatTensor(X[i:i+5]).to(device)
                y_captcha = y[i:i+5]
                outputs = model(x_captcha)
                preds = outputs.argmax(dim=1).cpu().numpy()
                if np.array_equal(preds, y_captcha):
                    correct += 1
                total += 1
    
    accuracy = 100 * correct / total if total > 0 else 0
    print(f'Full captcha accuracy: {accuracy:.2f}% ({correct}/{total})')
    return accuracy

if __name__ == "__main__":
    input_dir = "input"
    output_dir = "output"
    
    # Check if directories exist
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found!")
        exit(1)
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' not found!")
        exit(1)

    print("Loading and augmenting data...")
    X, y = load_data(input_dir, output_dir)
    
    if len(X) == 0:
        print("Error: No valid training data found!")
        exit(1)
    
    print(f"Loaded {len(X)} training samples")
    print(f"Number of classes: {len(CHARS)}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create data loaders
    train_dataset = CaptchaDataset(X_train, y_train, augment=True)
    val_dataset = CaptchaDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("Building improved model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ImprovedCaptchaModel(len(CHARS)).to(device)

    print("Training improved model...")
    train_losses, val_accuracies = train_model(model, train_loader, val_loader, num_epochs=50, device=device)

    print("Evaluating full captcha accuracy...")
    evaluate_full_captcha_accuracy(model, X_val, y_val, device=device)

    print("Saving final model...")
    torch.save(model.state_dict(), "captcha_model.pth")
    print("Model saved to captcha_model.pth")
    
    print("Training completed!")
