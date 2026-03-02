import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import numpy as np
import os
import sys
import cv2
from skimage.morphology import skeletonize

# Add parent directory to path for root imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import CrackClassifier, ResNetUNet, CrackRegressor, SeverityClassifier

class SoilCrackPipeline:
    def __init__(self, weights_dir='weights'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Models
        self.classifier = CrackClassifier().to(self.device)
        self.segmentor = ResNetUNet().to(self.device)
        self.regressor = CrackRegressor(input_dim=5).to(self.device)
        self.severity_classifier = SeverityClassifier().to(self.device)
        
        # Load Weights
        self._load_weights(weights_dir)
        
        self.classifier.eval()
        self.segmentor.eval()
        self.regressor.eval()
        self.severity_classifier.eval()

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def _load_weights(self, weights_dir):
        paths = {
            'classifier': os.path.join(weights_dir, 'stage1_best.pth'),
            'segmentor': os.path.join(weights_dir, 'stage2_best.pth'),
            'regressor': os.path.join(weights_dir, 'regressor_best_pixels.pth'),
            'severity': os.path.join(weights_dir, 'severity_best.pth'),
            'scaling': os.path.join(weights_dir, 'regressor_scaling.pth')
        }
        
        if os.path.exists(paths['classifier']):
            self.classifier.load_state_dict(torch.load(paths['classifier'], map_location=self.device))
        
        if os.path.exists(paths['segmentor']):
            self.segmentor.load_state_dict(torch.load(paths['segmentor'], map_location=self.device))
            
        if os.path.exists(paths['regressor']):
            self.regressor.load_state_dict(torch.load(paths['regressor'], map_location=self.device))
            
        if os.path.exists(paths['severity']):
            self.severity_classifier.load_state_dict(torch.load(paths['severity'], map_location=self.device))

        if os.path.exists(paths['scaling']):
            self.scaling = torch.load(paths['scaling'], map_location=self.device)
        else:
            self.scaling = {'min': [0.0]*5, 'max': [1.0]*5}

    def preprocess_image(self, image_path):
        """Preprocessing Pipeline: Resize only (Normalization happens in run)"""
        image_pil = Image.open(image_path).convert('RGB')
        # We return the PIL image to resize easily for both 224 (classifier) and 256 (segmentor)
        return image_pil

    def extract_mask_features(self, mask_np):
        mask = (mask_np > 0.5).astype(np.uint8)
        area = float(np.sum(mask))
        if area == 0: return [0.0] * 5
        
        coords = np.column_stack(np.where(mask > 0))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        bbox_w = float(x_max - x_min + 1)
        bbox_h = float(y_max - y_min + 1)
        density = area / (256 * 256)
        skeleton = skeletonize(mask > 0)
        length = float(np.sum(skeleton))
        
        features = [area, bbox_w, bbox_h, density, length]
        
        # Normalize features
        feat_min = torch.tensor(self.scaling['min'])
        feat_max = torch.tensor(self.scaling['max'])
        features = (torch.tensor(features) - feat_min) / (feat_max - feat_min + 1e-6)
        
        return features.unsqueeze(0).to(self.device)

    def generate_highlight(self, original_img, mask_np):
        """Overlay crack mask in semi-transparent red on the original resolution image"""
        # Resize mask to match original image dimensions
        mask_full = cv2.resize(mask_np, original_img.size, interpolation=cv2.INTER_NEAREST)
        mask_full_color = (mask_full > 0.5).astype(np.uint8) * 255
        
        highlight = np.array(original_img)
        
        # Create a red overlay at original resolution
        red_mask = np.zeros_like(highlight)
        red_mask[:, :, 0] = mask_full_color # Red channel
        
        # Overlay with 0.5 alpha
        highlight_ready = cv2.addWeighted(highlight, 1.0, red_mask, 0.5, 0)
        return Image.fromarray(highlight_ready)

    def run(self, image_path):
        # Apply preprocessing
        preprocessed_pil = self.preprocess_image(image_path)
        
        # Stage 1: Classification
        img_resized_class = preprocessed_pil.resize((224, 224))
        img_ts = self.to_tensor(img_resized_class)
        img_ts = self.norm(img_ts)
        img_ts = img_ts.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier(img_ts)
            # Debug prints
            print("Classification outputs:", outputs)
            _, predicted = outputs.max(1)
            print("Predicted class:", predicted.item())
            is_crack = predicted.item() == 1

        if not is_crack:
            recommendation = [
                "No visible soil cracks were detected in the analyzed region.",
                "Maintain current irrigation practices to preserve soil moisture balance.",
                "Continue routine monitoring during dry seasons to prevent crack formation.",
                "Ensure proper drainage to avoid sudden soil shrinkage."
            ]
            return {"status": "No Crack Detected", "crack_found": False, "recommendation": recommendation}

        # Stage 2: Segmentation
        img_resized_seg = preprocessed_pil.resize((256, 256))
        img_ts_seg = self.to_tensor(img_resized_seg)
        img_ts_seg = self.norm(img_ts_seg)
        img_ts_seg = img_ts_seg.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mask_logits = self.segmentor(img_ts_seg)
            # Debug prints for segmentation
            print("Max prediction value (logits):", mask_logits.max().item())
            mask_prob = torch.sigmoid(mask_logits)
            print("Max segmentation probability:", mask_prob.max().item())
            
            mask_binary = (mask_prob > 0.3).float()
            mask_np = mask_binary.squeeze().cpu().numpy()
            print("Mask sum:", mask_np.sum().item())

        # Crack Presence Check (threshold)
        if np.sum(mask_np) < 50: # Small threshold for noise
            recommendation = [
                "No visible soil cracks were detected in the analyzed region.",
                "Maintain current irrigation practices to preserve soil moisture balance.",
                "Continue routine monitoring during dry seasons to prevent crack formation.",
                "Ensure proper drainage to avoid sudden soil shrinkage."
            ]
            return {"status": "No Crack Detected", "crack_found": False, "recommendation": recommendation}

        # Stage 3: Regression (Pixel-based)
        features = self.extract_mask_features(mask_np)
        with torch.no_grad():
            metrics = self.regressor(features)
            length, width, area = metrics[0].tolist()

        # Highlighting
        highlight_img = self.generate_highlight(preprocessed_pil, mask_np)

        # Severity Prediction (FCNN-based)
        density = area / (256 * 256)
        severity_features = torch.tensor(
            [[length, width, area, density]], 
            dtype=torch.float32
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.severity_classifier(severity_features)
            severity_class = torch.argmax(logits, dim=1).item()
            print(f"Severity Prediction (Class {severity_class}): {logits}")
            
        if severity_class == 0:
            severity = "Low"
            recommendation = [
                "Minor surface cracking is observed in localized areas.",
                "Light ploughing can help improve soil aeration and structure.",
                "Maintain consistent irrigation to prevent further drying.",
                "Apply organic matter or mulch to improve moisture retention.",
                "Monitor the area periodically for any increase in crack depth or spread."
            ]
        elif severity_class == 1:
            severity = "Moderate"
            recommendation = [
                "Moderate cracking indicates noticeable soil shrinkage and stress.",
                "Deep ploughing is recommended to loosen compacted layers.",
                "Review irrigation scheduling to maintain uniform moisture levels.",
                "Incorporate soil conditioners to improve structural stability.",
                "Inspect surrounding regions for early signs of crack expansion.",
                "Regular monitoring is advised to prevent progression to severe levels."
            ]
        else:
            severity = "High"
            recommendation = [
                "Severe soil cracking suggests significant moisture loss and structural instability.",
                "Immediate soil stabilization measures should be implemented.",
                "Increase moisture retention using organic amendments or mulching.",
                "Consider controlled irrigation cycles to restore soil balance gradually.",
                "Assess the affected region for deep subsurface cracking.",
                "Professional agricultural or soil assessment is strongly recommended.",
                "Continuous monitoring is required to prevent further degradation."
            ]

        # Scale mask back to original resolution for output
        mask_full = cv2.resize(mask_np, preprocessed_pil.size, interpolation=cv2.INTER_NEAREST)

        return {
            "status": "Crack Present",
            "crack_found": True,
            "mask": mask_full,
            "highlight": highlight_img,
            "length": float(abs(length)),
            "width": float(abs(width)),
            "area": float(abs(area)),
            "severity": severity,
            "recommendation": recommendation
        }
