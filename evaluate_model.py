import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from models import CrackClassifier, ResNetUNet
from dataset import ClassificationDataset, SegmentationDataset

def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    
    # Weights paths
    stage1_weights = os.path.join('weights', 'stage1_best.pth')
    stage2_weights = os.path.join('weights', 'stage2_best.pth')
    
    # ---------------------------------------------------------
    # 1. STAGE 1: CLASSIFICATION EVALUATION
    # ---------------------------------------------------------
    print("\nEvaluating Stage 1: Binary Classification...")
    if not os.path.exists(stage1_weights):
        print(f"Warning: Stage 1 weights not found at {stage1_weights}. Skipping classification.")
        stage1_results = None
    else:
        test_ds_cls = ClassificationDataset(root_dir, split='test')
        test_loader_cls = DataLoader(test_ds_cls, batch_size=8, shuffle=False)
        
        model_cls = CrackClassifier().to(device)
        model_cls.load_state_dict(torch.load(stage1_weights, map_location=device))
        model_cls.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader_cls:
                images = images.to(device)
                outputs = model_cls(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=['non_crack', 'crack'])
        
        stage1_results = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'confusion_matrix': cm,
            'report': report
        }

    # ---------------------------------------------------------
    # 2. STAGE 2: SEGMENTATION EVALUATION
    # ---------------------------------------------------------
    print("Evaluating Stage 2: ResNet-UNet Segmentation...")
    if not os.path.exists(stage2_weights):
        print(f"Warning: Stage 2 weights not found at {stage2_weights}. Skipping segmentation.")
        stage2_results = None
    else:
        test_ds_seg = SegmentationDataset(root_dir, split='test')
        test_loader_seg = DataLoader(test_ds_seg, batch_size=1, shuffle=False)
        
        model_seg = ResNetUNet(n_class=1).to(device)
        model_seg.load_state_dict(torch.load(stage2_weights, map_location=device))
        model_seg.eval()
        
        total_dice = 0.0
        total_iou = 0.0
        
        with torch.no_grad():
            for images, masks in test_loader_seg:
                images, masks = images.to(device), masks.to(device)
                outputs = model_seg(images)
                pred = (torch.sigmoid(outputs) > 0.5).float()
                
                tp = (pred * masks).sum().item()
                fp = (pred * (1 - masks)).sum().item()
                fn = ((1 - pred) * masks).sum().item()
                
                # Image-level logic for Dice/IoU
                # To handle "empty-empty" cases properly in an average, we set them to 1.0
                if masks.sum() == 0 and pred.sum() == 0:
                    dice = 1.0
                    iou = 1.0
                else:
                    dice = (2. * tp) / (2. * tp + fp + fn + 1e-7)
                    iou = tp / (tp + fp + fn + 1e-7)
                
                total_dice += dice
                total_iou += iou
        
        num_seg = len(test_ds_seg)
        stage2_results = {
            'dice': total_dice / num_seg,
            'iou': total_iou / num_seg
        }

    # ---------------------------------------------------------
    # 3. FINAL REPORTING
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("Model Performance on Test Dataset")
    print("="*40)
    
    if stage1_results:
        print(f"Accuracy   : {stage1_results['accuracy']*100:.2f} %")
        print(f"Precision  : {stage1_results['precision']*100:.2f} %")
        print(f"Recall     : {stage1_results['recall']*100:.2f} %")
        print(f"F1 Score   : {stage1_results['f1']*100:.2f} %")
    else:
        print("Classification Metrics: Not Available")

    if stage2_results:
        print(f"Dice Score : {stage2_results['dice']*100:.2f} %")
        print(f"IoU        : {stage2_results['iou']*100:.2f} %")
    else:
        print("Segmentation Metrics: Not Available")
    print("-" * 40)
    
    if stage1_results:
        print("\nClassification Confusion Matrix:")
        print(stage1_results['confusion_matrix'])
        print("\nDetailed Classification Report:")
        print(stage1_results['report'])

if __name__ == "__main__":
    evaluate_model()
