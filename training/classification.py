import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Add parent directory to path for root imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import CrackClassifier
from dataset import ClassificationDataset

def train_stage1():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    lr = 0.0001
    batch_size = 32
    root_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset') # Corrected path
    weights_dir = os.path.join(os.path.dirname(__file__), '..', 'weights') # Corrected path
    
    # Dataset & Loader
    train_dataset = ClassificationDataset(root_dir, split='train')
    val_dataset = ClassificationDataset(root_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = CrackClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Early Stopping Config
    patience = 10
    no_improve_epochs = 0
    best_val_acc = 0.0
    
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    print(f"Starting Stage 1 Training on {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f}')
        
        # Early Stopping & Best Model Saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(weights_dir, 'stage1_best.pth'))
            print(f'--> Saved Best Model with Acc: {best_val_acc:.2f}%')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

if __name__ == "__main__":
    train_stage1()
