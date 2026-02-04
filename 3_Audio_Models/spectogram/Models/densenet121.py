import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class DenseNet121Binary(nn.Module):
    """
    DenseNet121 adapted for BINARY classification with a custom classifier head.

    Key characteristics:
    1. Standard RGB (3-channel) input
    2. Frozen pretrained DenseNet121 backbone
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

        # 1. Load DenseNet121 backbone (RGB by default)
        if pretrained:
            self.densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)
        else:
            self.densenet = densenet121(weights=None)

        # 2. Freeze DenseNet backbone
        for param in self.densenet.parameters():
            param.requires_grad = False

        # 3. Build improved classifier head
        classifier_layers = []
        in_features = self.densenet.classifier.in_features  # 1024 for DenseNet121

        for hidden_size in hidden_units:
            classifier_layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ])
            in_features = hidden_size

        # Final binary output
        classifier_layers.append(nn.Linear(in_features, 1))

        self.densenet.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Expected input shape:
            (batch_size, 3, H, W)  — RGB images

        Returns:
            (batch_size, 1) logits
        """
        return self.densenet(x)
