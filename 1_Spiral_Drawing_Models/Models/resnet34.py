import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from torch import inference_mode


class ResNet34Binary(nn.Module):
    """
    ResNet34 adapted for BINARY classification with a custom classifier head
    and grayscale input (1 channel) using RGB weight averaging.

    Key improvements:
    1. Grayscale input (1 channel)
    2. Frozen pretrained backbone
    3. Multi-layer classifier head with dropout
    4. SINGLE output neuron for binary classification
    """

    def __init__(self, dropout_rate=0.5, hidden_units=[512, 128], pretrained=True):
        super().__init__()

        # 1. Load ResNet34
        if pretrained:
            self.resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        else:
            self.resnet = resnet34(weights=None)

        # 2. Modify first conv layer for grayscale input
        old_conv = self.resnet.conv1
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        # 3. Copy pretrained weights averaged across RGB channels
        with inference_mode():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias[:] = old_conv.bias

        # 4. Replace first conv
        self.resnet.conv1 = new_conv

        # 5. Freeze backbone
        for param in self.resnet.parameters():
            param.requires_grad = False

        # 6. Build improved classifier
        classifier_layers = []
        in_features = self.resnet.fc.in_features  # 512 for ResNet34

        for hidden_size in hidden_units:
            classifier_layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            in_features = hidden_size

        # Final binary output
        classifier_layers.append(nn.Linear(in_features, 1))
        self.resnet.fc = nn.Sequential(*classifier_layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, 1, H, W) — grayscale images

        Returns:
            (batch, 1) logits
        """
        return self.resnet(x)
