import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path for root imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import SeverityClassifier

class CSVDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        # features: length, width, area, density
        self.features = torch.tensor(df.iloc[:, :4].values, dtype=torch.float32)
        self.labels = torch.tensor(df.iloc[:, 4].values, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_severity():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    lr = 0.001
    batch_size = 32
    
    weights_dir = os.path.join(os.path.dirname(__file__), '..', 'weights')
    csv_file = os.path.join(os.path.dirname(__file__), 'severity_dataset.csv')
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run generate_severity_dataset.py first.")
        return

    dataset = CSVDataset(csv_file)
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = SeverityClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    best_val_acc = 0.0
    print(f"Starting Severity Training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
            
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(weights_dir, 'severity_best.pth'))

    print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")
    print(f"Model saved to weights/severity_best.pth")

if __name__ == "__main__":
    train_severity()
