import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import sys
import numpy as np
import cv2
from skimage.morphology import skeletonize
from tqdm import tqdm

# Add parent directory to path for root imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import CrackRegressor
from dataset import SegmentationDataset

def extract_features_and_targets(mask_np):
    """
    Extracts features for regressor input and targets for training.
    Features: [area, bbox_w, bbox_h, density, skeleton_length]
    Targets: [length, width, area]
    """
    mask = (mask_np > 0.5).astype(np.uint8)
    area = float(np.sum(mask))
    
    if area == 0:
        return [0.0] * 5, [0.0] * 3

    # Bounding Box
    coords = np.column_stack(np.where(mask > 0))
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    bbox_w = float(x_max - x_min + 1)
    bbox_h = float(y_max - y_min + 1)
    
    # Density
    density = area / (256 * 256)
    
    # Skeletonization for length
    skeleton = skeletonize(mask > 0)
    length = float(np.sum(skeleton))
    
    # Width = Area / Length (if length > 0)
    width = area / length if length > 0 else 0.0
    
    features = [area, bbox_w, bbox_h, density, length]
    targets = [length, width, area]
    
    return features, targets

class PixelRegressionDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
        # Simple Min-Max scaling for features (pre-computed for inference)
        self.feature_min = self.features.min(dim=0)[0]
        self.feature_max = self.features.max(dim=0)[0]
        # Avoid division by zero
        self.feature_max[self.feature_max == self.feature_min] += 1e-6
        
        self.features = (self.features - self.feature_min) / (self.feature_max - self.feature_min)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def train_regressor():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    weights_dir = os.path.join(os.path.dirname(__file__), '..', 'weights')
    
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    print("Loading segmentation masks and extracting features...")
    seg_dataset = SegmentationDataset(root_dir, split='train')
    
    all_features = []
    all_targets = []
    
    for i in tqdm(range(len(seg_dataset))):
        _, mask_ts = seg_dataset[i]
        mask_np = mask_ts.squeeze().numpy()
        feat, target = extract_features_and_targets(mask_np)
        if target[2] > 0: # Only train on samples with cracks
            all_features.append(feat)
            all_targets.append(target)

    if not all_features:
        print("No samples with cracks found in dataset. Using dummy data for architecture verification.")
        all_features = np.random.rand(10, 5).tolist()
        all_targets = np.random.rand(10, 3).tolist()

    dataset = PixelRegressionDataset(all_features, all_targets)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = CrackRegressor(input_dim=5).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    patience = 15
    best_loss = float('inf')
    counter = 0

    print(f"Training Pixel Regressor on {device}...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(weights_dir, 'regressor_best_pixels.pth'))
            # Save feature scaling params for inference
            scaling_params = {
                'min': dataset.feature_min.tolist(),
                'max': dataset.feature_max.tolist()
            }
            torch.save(scaling_params, os.path.join(weights_dir, 'regressor_scaling.pth'))
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Training complete. Best Loss:", best_loss)

if __name__ == "__main__":
    train_regressor()
