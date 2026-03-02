import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Add parent directory to path for root imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import ResNetUNet
from dataset import SegmentationDataset

def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def train_stage2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    lr = 0.0001
    batch_size = 16
    root_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    weights_dir = os.path.join(os.path.dirname(__file__), '..', 'weights')
    
    train_dataset = SegmentationDataset(root_dir, split='train')
    val_dataset = SegmentationDataset(root_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = ResNetUNet(n_class=1).to(device)
    bce_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)

    # Check if we should resume from a full checkpoint
    checkpoint_path = os.path.join(weights_dir, 'stage2_checkpoint.pth')
    start_epoch = 0
    best_val_dice = 0.0

    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_dice = checkpoint['best_val_dice']
        print(f"Continuing from Epoch {start_epoch} with Best Dice: {best_val_dice:.4f}")
    elif os.path.exists(os.path.join(weights_dir, 'stage2_best.pth')):
        # Fallback if only the best model exists
        print("Loading best model weights to resume...")
        model.load_state_dict(torch.load(os.path.join(weights_dir, 'stage2_best.pth'), map_location=device))
        
    # Early Stopping Config
    patience = 15
    no_improve_epochs = 0
    
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    print(f"Starting Stage 2 Training on {device}...")
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss_bce = bce_criterion(outputs, masks)
            loss_dice = dice_loss(outputs, masks)
            loss = loss_bce + loss_dice 
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Validation
        model.eval()
        total_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                pred = (torch.sigmoid(outputs) > 0.5).float()
                
                for p, m in zip(pred, masks):
                    p_sum = p.sum()
                    m_sum = m.sum()
                    if p_sum == 0 and m_sum == 0:
                        total_dice += 1.0 
                    else:
                        intersection = (p * m).sum()
                        dice = (2. * intersection) / (p_sum + m_sum + 1e-7)
                        total_dice += dice.item()
        
        avg_dice = total_dice / len(val_dataset)
        print(f'Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f} | Val Dice: {avg_dice:.4f}')
        
        scheduler.step(avg_dice)
        
        # Save Checkpoint for True Resume
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_dice': best_val_dice
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Save Best Model
        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            torch.save(model.state_dict(), os.path.join(weights_dir, 'stage2_best.pth'))
            print(f'--> Saved Best Model with Dice: {best_val_dice:.4f}')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

if __name__ == "__main__":
    train_stage2()
