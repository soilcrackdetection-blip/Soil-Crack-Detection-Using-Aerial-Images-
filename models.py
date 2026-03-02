import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class CrackClassifier(nn.Module):
    """Stage 1: Binary Classification (Crack vs Non-Crack)"""
    def __init__(self, model_name='resnet18', pretrained=True):
        super(CrackClassifier, self).__init__()
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        else:
            self.model = models.resnet34(pretrained=pretrained)
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.model(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ResNetUNet(nn.Module):
    """Stage 2: ResNet-UNet Segmentation"""
    def __init__(self, n_class=1):
        super().__init__()
        self.base_model = models.resnet34(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = nn.Sequential(*self.base_layers[3:4]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1 = nn.Sequential(*self.base_layers[4]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = nn.Sequential(*self.base_layers[5]) # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = nn.Sequential(*self.base_layers[6]) # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = nn.Sequential(*self.base_layers[7]) # size=(N, 512, x.H/32, x.W/32)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up3 = DoubleConv(512 + 256, 256)
        self.up2 = DoubleConv(256 + 128, 128)
        self.up1 = DoubleConv(128 + 64, 64)
        self.up0 = DoubleConv(64 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_orig = input
        layer0 = self.layer0(input)
        layer0_1x1 = self.layer0_1x1(layer0)
        layer1 = self.layer1(layer0_1x1)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.upsample(layer4)
        x = torch.cat([x, layer3], dim=1)
        x = self.up3(x)

        x = self.upsample(x)
        x = torch.cat([x, layer2], dim=1)
        x = self.up2(x)

        x = self.upsample(x)
        x = torch.cat([x, layer1], dim=1)
        x = self.up1(x)

        x = self.upsample(x)
        x = torch.cat([x, layer0], dim=1)
        x = self.up0(x)

        x = self.upsample(x)
        out = self.conv_last(x)

        return out

class CrackRegressor(nn.Module):
    """Stage 3: FFN Regression for Length, Width, Area (Pixels)"""
    def __init__(self, input_dim=5):
        super(CrackRegressor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # Predicted length, width, area (all pixels)
        )

    def forward(self, x):
        return self.fc(x)

class SeverityClassifier(nn.Module):
    """Stage 4: FCNN for Severity Classification (Low, Moderate, High)"""
    def __init__(self):
        super(SeverityClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3) # Low, Moderate, High
        )

    def forward(self, x):
        return self.fc(x)
