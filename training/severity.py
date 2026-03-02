import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import numpy as np

# Add parent directory to path for root imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import SeverityClassifier
from dataset import SeverityDataset

def train_stage4():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    lr = 0.0001
    batch_size = 32
    
    weights_dir = os.path.join(os.path.dirname(__file__), '..', 'weights')
    
    # Dummy data for architecture setup
    # Features: [Length, Width, Area]
    # Labels: 0 (Low), 1 (Moderate), 2 (High)
    dummy_features = np.random.rand(200, 3)
    dummy_labels = np.random.randint(0, 3, 200)
    
    dataset = SeverityDataset(dummy_features, dummy_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SeverityClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Early Stopping Config
    patience = 10
    no_improve_epochs = 0
    best_val_acc = 0.0
    
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    print(f"Starting Stage 4 Training on {device}...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        acc = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.4f} | Acc: {acc:.2f}%')
        
        # Early Stopping & Best Model Saving (based on Accuracy)
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), os.path.join(weights_dir, 'stage4_best.pth'))
            print(f'--> Saved Best Model with Acc: {best_val_acc:.2f}%')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

if __name__ == "__main__":
    train_stage4()
