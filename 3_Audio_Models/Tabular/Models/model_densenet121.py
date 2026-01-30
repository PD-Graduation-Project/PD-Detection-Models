import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet1D(nn.Module):
    """
    Adapts DenseNet (designed for 2D images) to work with 1D tabular data.
    
    Necessary Changes:
    1. Convert input (batch, 19) -> (batch, 1, 19, 1) for 2D convolutions
    2. Replace first conv layer to accept 1-channel input
    3. Modify pooling layers to prevent dimension collapse
    4. Replace final classifier for binary output
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load DenseNet121
        if pretrained:
            self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        else:
            self.densenet = models.densenet121(weights=None)
        
        # CHANGE 1: Replace first conv layer
        # Original: Conv2d(3, 64, kernel_size=7) for RGB images
        # New: Conv2d(1, 64, kernel_size=(7,1)) for 1-channel input
        self.densenet.features.conv0 = nn.Conv2d(
            1, 64, 
            kernel_size=(7, 1),
            stride=(1, 1),  # Changed from (2,1) to (1,1) to preserve size
            padding=(3, 0), 
            bias=False
        )
        
        # CHANGE 2: Modify first pooling layer to only pool along height dimension
        self.densenet.features.pool0 = nn.MaxPool2d(
            kernel_size=(2, 1),  # Only pool height, not width
            stride=(2, 1),
            padding=0
        )
        
        # CHANGE 3: Modify transition layers to prevent width collapse
        # Transition1
        self.densenet.features.transition1.pool = nn.AvgPool2d(
            kernel_size=(2, 1),
            stride=(2, 1)
        )
        # Transition2
        self.densenet.features.transition2.pool = nn.AvgPool2d(
            kernel_size=(2, 1),
            stride=(2, 1)
        )
        # Transition3
        self.densenet.features.transition3.pool = nn.AvgPool2d(
            kernel_size=(2, 1),
            stride=(2, 1)
        )
        
        # CHANGE 4: Replace final classifier
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, 1)
        
    def forward(self, x):
        # Input: (batch, 19) from your dataloader
        
        # Reshape: (batch, 19) -> (batch, 1, 19, 1)
        # This makes it "image-like" for 2D convolutions
        x = x.unsqueeze(1).unsqueeze(-1)
        
        # Forward through DenseNet
        x = self.densenet(x)
        
        return x