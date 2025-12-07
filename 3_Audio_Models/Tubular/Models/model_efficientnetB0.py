import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNet1D(nn.Module):
    """
    Adapts EfficientNet (designed for 2D images) to work with 1D tabular data.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load EfficientNet-B0 (you can use b1, b2, ..., b7)
        if pretrained:
            self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        else:
            self.efficientnet = models.efficientnet_b0(weights=None)
        
        # CHANGE 1: Replace first conv layer
        self.efficientnet.features[0][0] = nn.Conv2d(
            1, 32,
            kernel_size=(3, 1),
            stride=(1, 1),  # Changed to preserve size
            padding=(1, 0),
            bias=False
        )
        
        # CHANGE 2: Modify MBConv blocks to prevent width collapse
        # This is complex, so we use adaptive pooling at the end instead
        
        # CHANGE 3: Replace final classifier
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 1)
        )
        
        # Add adaptive pooling before classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        # Reshape: (batch, 19) -> (batch, 1, 19, 1)
        x = x.unsqueeze(1).unsqueeze(-1)
        
        # Forward through features
        x = self.efficientnet.features(x)
        
        # Adaptive pooling to ensure correct size
        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.efficientnet.classifier(x)
        
        return x