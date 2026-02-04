import torch
import torch.nn as nn
import torchvision.models as models

class MobileNet1D_V3(nn.Module):
    """
    Adapts MobileNetV3 (designed for 2D images) to work with 1D tabular data.
    Lightweight and fast!
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load MobileNetV3
        if pretrained:
            self.mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        else:
            self.mobilenet = models.mobilenet_v3_large(weights=None)
        
        # CRITICAL FIX: MobileNetV3's first layer outputs 16 channels, NOT 32
        # We need to replace it completely with correct output channels
        
        # Original first layer structure in MobileNetV3:
        # features[0] is a Sequential containing:
        #   [0]: Conv2d(3, 16, kernel_size=3, stride=2)
        #   [1]: BatchNorm2d(16)
        #   [2]: Hardswish()
        
        # CHANGE 1: Replace the entire first block
        self.mobilenet.features[0] = nn.Sequential(
            nn.Conv2d(
                1, 16,  # Input: 1 channel, Output: 16 channels (match original)
                kernel_size=(3, 1),
                stride=(1, 1),  # Changed to preserve size
                padding=(1, 0),
                bias=False
            ),
            nn.BatchNorm2d(16),  # Must match the 16 output channels
            nn.Hardswish()
        )
        
        # CHANGE 2: Add adaptive pooling to handle variable spatial dimensions
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # CHANGE 3: Replace final classifier
        # MobileNetV3 has a more complex classifier structure
        # classifier[0] is Linear(960, 1280)
        # classifier[3] is Linear(1280, num_classes)
        in_features = self.mobilenet.classifier[0].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1280, 1)
        )
        
    def forward(self, x):
        # Input: (batch, 19) from your dataloader
        
        # Reshape: (batch, 19) -> (batch, 1, 19, 1)
        x = x.unsqueeze(1).unsqueeze(-1)
        
        # Forward through features
        x = self.mobilenet.features(x)
        
        # Adaptive pooling to ensure (batch, channels, 1, 1)
        x = self.gap(x)
        
        # Flatten to (batch, channels)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.mobilenet.classifier(x)
        
        return x