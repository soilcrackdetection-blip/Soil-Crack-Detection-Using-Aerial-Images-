import torch
import os
import sys
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.inference import SoilCrackPipeline

def verify_severity():
    pipeline = SoilCrackPipeline()
    test_images_dir = os.path.join('dataset', 'segmentation', 'test', 'images')
    
    # Try a few images
    test_images = ['crack_3.jpg', 'crack_14.jpg']
    
    for img_name in test_images:
        img_path = os.path.join(test_images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found.")
            continue
            
        print(f"\n--- Testing Image: {img_name} ---")
        result = pipeline.run(img_path)
        
        if result['status'] == "Crack Present":
            print(f"Predicted Geometric Metrics: L={result['length']:.2f}, W={result['width']:.2f}, A={result['area']:.2f}")
            print(f"Final Severity: {result['severity']}")
            print(f"Recommendation: {result['recommendation']}")
        else:
            print(f"Status: {result['status']}")

if __name__ == "__main__":
    verify_severity()
