import torch
import os
import sys
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import ResNetUNet
from dataset import SegmentationDataset

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    weights_path = os.path.join(os.path.dirname(__file__), '..', 'weights', 'stage2_best.pth')
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}. Finish training first!")
        return

    # Load Test Dataset
    test_dataset = SegmentationDataset(root_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Load Model
    model = ResNetUNet(n_class=1).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    print(f"Evaluating Stage 2 on {len(test_dataset)} test images...")
    
    total_dice = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            pred = (torch.sigmoid(outputs) > 0.5).float()
            
            p_sum = pred.sum()
            m_sum = masks.sum()
            if p_sum == 0 and m_sum == 0:
                total_dice += 1.0 # Perfect non-crack detection
            else:
                intersection = (pred * masks).sum()
                dice = (2. * intersection) / (p_sum + m_sum + 1e-7)
                total_dice += dice.item()
            
    avg_dice = total_dice / len(test_dataset)
    print("-" * 30)
    print(f"FINAL TEST DICE SCORE: {avg_dice:.4f}")
    print(f"APPROX ACCURACY: {avg_dice * 100:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    evaluate()
