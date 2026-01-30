import torch
import torch.nn as nn
import torchvision.models as models

class MobileNet1D_V2(nn.Module):
    """
    Adapts MobileNetV2 (designed for 2D images) to work with 1D tabular data.
    Lightweight and fast!
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load MobileNetV2
        if pretrained:
            self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            self.mobilenet = models.mobilenet_v2(weights=None)
        
        # CHANGE 1: Replace first conv layer
        self.mobilenet.features[0][0] = nn.Conv2d(
            1, 32,
            kernel_size=(3, 1),
            stride=(1, 1),  # Changed to preserve size
            padding=(1, 0),
            bias=False
        )
        
        # CHANGE 2: Add adaptive pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # CHANGE 3: Replace final classifier
        self.mobilenet.classifier[1] = nn.Linear(1280, 1)
        
    def forward(self, x):
        # Reshape: (batch, 19) -> (batch, 1, 19, 1)
        x = x.unsqueeze(1).unsqueeze(-1)
        
        # Forward through features
        x = self.mobilenet.features(x)
        
        # Adaptive pooling
        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.mobilenet.classifier(x)
        
        return x