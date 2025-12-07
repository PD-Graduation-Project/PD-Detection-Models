import torch
import torch.nn as nn
import torchvision.models as models

class ResNet1D(nn.Module):
    """
    Adapts ResNet (designed for 2D images) to work with 1D tabular data.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load ResNet18
        if pretrained:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.resnet = models.resnet18(weights=None)
        
        # CHANGE 1: Replace first conv layer
        self.resnet.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=(7, 1),
            stride=(1, 1),  # Changed from (2,1) to preserve size
            padding=(3, 0),
            bias=False
        )
        
        # CHANGE 2: Modify maxpool to only pool height
        self.resnet.maxpool = nn.MaxPool2d(
            kernel_size=(2, 1),
            stride=(2, 1),
            padding=0
        )
        
        # CHANGE 3: Modify downsampling in residual blocks
        # Layer2 downsample
        if hasattr(self.resnet.layer2[0], 'downsample') and self.resnet.layer2[0].downsample is not None:
            self.resnet.layer2[0].downsample[0] = nn.Conv2d(
                64, 128,
                kernel_size=(1, 1),
                stride=(2, 1),  # Only stride on height
                bias=False
            )
        
        # Layer3 downsample
        if hasattr(self.resnet.layer3[0], 'downsample') and self.resnet.layer3[0].downsample is not None:
            self.resnet.layer3[0].downsample[0] = nn.Conv2d(
                128, 256,
                kernel_size=(1, 1),
                stride=(2, 1),
                bias=False
            )
        
        # Layer4 downsample
        if hasattr(self.resnet.layer4[0], 'downsample') and self.resnet.layer4[0].downsample is not None:
            self.resnet.layer4[0].downsample[0] = nn.Conv2d(
                256, 512,
                kernel_size=(1, 1),
                stride=(2, 1),
                bias=False
            )
        
        # CHANGE 4: Replace final FC layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)
        
    def forward(self, x):
        # Reshape: (batch, 19) -> (batch, 1, 19, 1)
        x = x.unsqueeze(1).unsqueeze(-1)
        
        # Forward through ResNet
        x = self.resnet(x)
        
        return x