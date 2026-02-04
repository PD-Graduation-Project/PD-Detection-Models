import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Binary(nn.Module):
    """
    ResNet18 adapted for BINARY classification with a custom classifier head.

    Key characteristics:
    1. Standard RGB (3-channel) input
    2. Frozen pretrained ResNet18 backbone
    3. Multi-layer classifier head with dropout
    4. SINGLE output neuron (for BCEWithLogitsLoss)

    Args:
        dropout_rate (float): Dropout probability (default: 0.5)
        hidden_units (list[int]): Hidden layer sizes for classifier head
        pretrained (bool): Whether to load ImageNet pretrained weights
    """

    def __init__(
        self,
        dropout_rate: float = 0.5,
        hidden_units: list[int] = [512, 128],
        pretrained: bool = True,
    ):
        super().__init__()

        # 1. Load ResNet18 backbone (RGB by default)
        if pretrained:
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.resnet = resnet18(weights=None)

        # 2. Freeze all pretrained parameters
        for param in self.resnet.parameters():
            param.requires_grad = False

        # 3. Build improved classifier head
        classifier_layers = []
        in_features = self.resnet.fc.in_features  # 512 for ResNet18

        for hidden_size in hidden_units:
            classifier_layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ])
            in_features = hidden_size

        # Final binary output
        classifier_layers.append(nn.Linear(in_features, 1))

        self.resnet.fc = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Expected input shape:
            (batch_size, 3, H, W) — RGB images

        Returns:
            (batch_size, 1) logits
        """
        return self.resnet(x)
