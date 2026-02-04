import torch
import torch.nn as nn
from torchvision.models import densenet161
from torch import inference_mode


class DenseNet161Binary(nn.Module):
    """
    DenseNet161 adapted for BINARY classification with a custom classifier head
    and grayscale input (1 channel) using RGB weight averaging.

    Key improvements:
    1. Multi-layer classifier head with dropout
    2. Global average pooling
    3. SINGLE output neuron for binary classification
    4. Frozen backbone by default

    Args:
        dropout_rate (float): Dropout probability (default 0.5)
        hidden_units (list[int]): Hidden layer sizes for classifier head
        pretrained (bool): Whether to load pretrained ImageNet weights
    """

    def __init__(self, dropout_rate=0.5, hidden_units=[512, 128], pretrained=True):
        super().__init__()

        # 1. Load DenseNet161
        if pretrained:
            self.densenet = densenet161(weights="DEFAULT")
        else:
            self.densenet = densenet161(weights=None)

        # 2. Get original input conv
        old_conv = self.densenet.features[0]

        # 3. Create new conv layer for grayscale input
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        # 4. Copy pretrained weights averaged across RGB channels
        with inference_mode():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias[:] = old_conv.bias

        # 5. Replace first conv layer
        self.densenet.features[0] = new_conv

        # 6. Freeze backbone
        for param in self.densenet.parameters():
            param.requires_grad = False

        # 7. Build improved classifier
        classifier_layers = []
        in_features = self.densenet.classifier.in_features  # 2208 for DenseNet161

        for hidden_size in hidden_units:
            classifier_layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_features = hidden_size

        # Final binary output
        classifier_layers.append(nn.Linear(in_features, 1))
        self.densenet.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, 1, H, W) — grayscale images

        Returns:
            (batch, 1) logits
        """
        return self.densenet(x)
