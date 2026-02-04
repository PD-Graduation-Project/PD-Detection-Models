import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_small_Weights


class MobileNetV3SmallBinary(nn.Module):
    """
    MobileNetV3-Small adapted for BINARY classification with a custom classifier head.

    Key characteristics:
    1. Standard RGB (3-channel) input
    2. Frozen pretrained MobileNetV3-Small backbone
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

        # 1. Load MobileNetV3-Small backbone (RGB by default)
        if pretrained:
            self.mobilenet = mobilenet_v3_small(
                weights=MobileNet_V3_small_Weights.DEFAULT
            )
        else:
            self.mobilenet = mobilenet_v3_small(weights=None)

        # 2. Freeze all pretrained parameters
        for param in self.mobilenet.parameters():
            param.requires_grad = False

        # 3. Build improved classifier head
        classifier_layers = []
        in_features = self.mobilenet.classifier[3].in_features  # 1280

        for hidden_size in hidden_units:
            classifier_layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ])
            in_features = hidden_size

        # Final binary output
        classifier_layers.append(nn.Linear(in_features, 1))

        self.mobilenet.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Expected input shape:
            (batch_size, 3, H, W) — RGB images

        Returns:
            (batch_size, 1) logits
        """
        return self.mobilenet(x)
