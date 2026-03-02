import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class ClassificationDataset(Dataset):
    """Dataset for Stage 1: Binary Classification"""
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, 'classification', split)
        self.classes = ['non_crack', 'crack'] # 0: non_crack, 1: crack
        self.image_paths = []
        self.labels = []

        for label, cls in enumerate(self.classes):
            cls_dir = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.labels.append(label)

        if transform:
            self.transform = transform
        else:
            # Standard ImageNet normalization
            norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            if split == 'train':
                # Images are already 224x224, so no Resize needed
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15), # Mild rotation (small angle)
                    transforms.ColorJitter(brightness=0.1, contrast=0.1), # Mild brightness/contrast variation
                    transforms.ToTensor(),
                    norm
                ])
            else:
                # No augmentation for val and test
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    norm
                ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class SegmentationDataset(Dataset):
    """Dataset for Stage 2: Segmentation"""
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, 'segmentation', split)
        self.imgs_dir = os.path.join(self.root_dir, 'images')
        self.masks_dir = os.path.join(self.root_dir, 'masks')
        self.img_names = os.listdir(self.imgs_dir)
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.imgs_dir, img_name)
        # Masks are .png while images are .jpg
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') # Gray scale

        # Apply physical augmentation only to TRAIN split
        if self.split == 'train':
            # Horizontal flip
            if torch.rand(1) > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            
            # Small rotation
            angle = float(torch.empty(1).uniform_(-10, 10).item())
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)

        # ToTensor
        image_ts = transforms.functional.to_tensor(image)
        mask_ts = transforms.functional.to_tensor(mask)

        # Normalize Image (ImageNet) only
        image_ts = transforms.functional.normalize(image_ts, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Masks must remain binary (0 and 1)
        mask_ts = (mask_ts > 0.5).float()

        return image_ts, mask_ts

class RegressionDataset(Dataset):
    """Dummy or specific dataset for Regression if needed, usually from masks"""
    def __init__(self, data_list):
        # data_list: [[mask_tensor, length, width, area], ...]
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mask, l, w, a = self.data[idx]
        return mask, torch.tensor([l, w, a], dtype=torch.float32)

class SeverityDataset(Dataset):
    """Dataset for Stage 4: Severity Classification"""
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
