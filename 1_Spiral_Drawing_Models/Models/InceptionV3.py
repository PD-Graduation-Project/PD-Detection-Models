import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from torch import inference_mode


class InceptionV3Binary(nn.Module):
    """
    InceptionV3 adapted for BINARY classification with a custom classifier head
    and grayscale input (1 channel) using RGB weight averaging.

    Key improvements:
    1. Grayscale input (1 channel)
    2. Frozen pretrained backbone
    3. Multi-layer classifier head with dropout
    4. SINGLE output neuron for binary classification
    5. Auxiliary logits disabled
    """

    def __init__(self, dropout_rate=0.5, hidden_units=[512, 128], pretrained=True):
        super().__init__()

        # 1. Load InceptionV3
        if pretrained:
            self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=False)
        else:
            self.inception = inception_v3(weights=None, aux_logits=False)

        # 2. Modify first conv layer for grayscale input
        old_conv = self.inception.Conv2d_1a_3x3.conv
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
        self.inception.Conv2d_1a_3x3.conv = new_conv

        # 5. Freeze backbone
        for param in self.inception.parameters():
            param.requires_grad = False

        # 6. Build improved classifier
        classifier_layers = []
        in_features = self.inception.fc.in_features  # 2048

        for hidden_size in hidden_units:
            classifier_layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            in_features = hidden_size

        # Final binary output
        classifier_layers.append(nn.Linear(in_features, 1))
        self.inception.fc = nn.Sequential(*classifier_layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, 1, H, W) — grayscale images

        Returns:
            (batch, 1) logits
        """
        return self.inception(x)
