import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.morphology import skeletonize
import sys

# Add parent directory to path for root imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def generate_dataset():
    root_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'segmentation')
    output_file = os.path.join(os.path.dirname(__file__), 'severity_dataset.csv')
    
    dataset = []
    
    for split in ['train', 'val', 'test']:
        mask_dir = os.path.join(root_dir, split, 'masks')
        if not os.path.exists(mask_dir):
            continue
            
        print(f"Processing {split} masks...")
        for mask_name in os.listdir(mask_dir):
            mask_path = os.path.join(mask_dir, mask_name)
            mask = np.array(Image.open(mask_path).convert('L'))
            mask = (mask > 127).astype(np.uint8)
            
            area = float(np.sum(mask))
            if area == 0:
                continue
                
            # Feature Extraction (matches inference/regressor)
            area_ratio = area / (256 * 256)
            
            # Skeleton for length
            skeleton = skeletonize(mask > 0)
            length = float(np.sum(skeleton))
            if length == 0: length = 1.0 # Avoid div zero
            
            width = area / length
            density = area_ratio # area / (256*256)
            
            # Severity Labeling (based on area_ratio)
            if area_ratio < 0.02:
                severity = 0   # Low
            elif area_ratio < 0.06:
                severity = 1   # Moderate
            else:
                severity = 2   # High
                
            dataset.append({
                'length': length,
                'width': width,
                'area': area,
                'density': density,
                'severity': severity
            })
            
    df = pd.DataFrame(dataset)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} samples to {output_file}")

if __name__ == "__main__":
    generate_dataset()
