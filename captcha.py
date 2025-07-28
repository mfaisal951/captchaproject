import os
import cv2
import numpy as np
import string
import torch
import torch.nn as nn
import pickle

class Captcha(object):
    def __init__(self, model_path='captcha_model.pth', mapping_path='char_mappings.pkl'):
        """
        Initialize the Captcha recognizer by loading the trained model and character mappings.
        """
        # Load character mappings
        with open(mapping_path, 'rb') as f:
            mappings = pickle.load(f)
            self.char_to_idx = mappings['char_to_idx']
            self.idx_to_char = mappings['idx_to_char']
        self.CHARS = string.ascii_uppercase + string.digits
        
        # Define the improved model architecture (must match training)
        class ImprovedCaptchaModel(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                
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
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ImprovedCaptchaModel(len(self.CHARS)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def preprocess(self, img):
        """
        Improved preprocessing with better normalization (matches training).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # Apply adaptive thresholding for better contrast
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        resized = cv2.resize(gray, (32, 32))  # Increased size for better detail
        norm = resized.astype(np.float32) / 255.0
        return norm.reshape(1, 32, 32)

    def split_captcha(self, img):
        """
        Improved character splitting with better boundaries (matches training).
        """
        h, w = img.shape[:2]
        char_width = w // 5
        # Add small overlap to avoid cutting characters
        overlap = char_width // 10
        return [img[:, max(0, i * char_width - overlap):min(w, (i + 1) * char_width + overlap)] 
                for i in range(5)]

    def __call__(self, im_path, save_path):
        """
        Perform inference on the given image and save the predicted text to save_path.
        Args:
            im_path: Path to the .jpg image to load and infer.
            save_path: Output file path to save the one-line outcome.
        """
        img = cv2.imread(im_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {im_path}")
        
        char_imgs = self.split_captcha(img)
        chars = []
        
        for char_img in char_imgs:
            x = self.preprocess(char_img)
            x = torch.FloatTensor(x).unsqueeze(0).to(self.device)  # shape: (1, 1, 32, 32)
            
            with torch.no_grad():
                output = self.model(x)
                pred_idx = output.argmax(dim=1).item()
                chars.append(self.idx_to_char[pred_idx])
        
        captcha = ''.join(chars)
        with open(save_path, 'w') as f:
            f.write(captcha + '\n')
        print(f"Predicted CAPTCHA: {captcha} (saved to {save_path})")
