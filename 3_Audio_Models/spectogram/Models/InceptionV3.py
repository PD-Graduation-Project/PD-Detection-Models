import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights


class InceptionV3Binary(nn.Module):
    """
    InceptionV3 adapted for BINARY classification with a custom classifier head.

    Key characteristics:
    1. Standard RGB (3-channel) input
    2. Frozen pretrained InceptionV3 backbone
    3. Multi-layer classifier head with dropout
    4. SINGLE output neuron (for BCEWithLogitsLoss)
    5. Auxiliary logits disabled for stable training

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

        # 1. Load InceptionV3 backbone (RGB by default, aux logits disabled)
        if pretrained:
            self.inception = inception_v3(
                weights=Inception_V3_Weights.DEFAULT,
                aux_logits=True,
            )
        else:
            self.inception = inception_v3(
                weights=None,
                aux_logits=False,
            )

        # 2. Freeze all pretrained parameters
        for param in self.inception.parameters():
            param.requires_grad = False

        # 3. Build improved classifier head
        classifier_layers = []
        in_features = self.inception.fc.in_features  # 2048

        for hidden_size in hidden_units:
            classifier_layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ])
            in_features = hidden_size

        # Final binary output layer
        classifier_layers.append(nn.Linear(in_features, 1))

        self.inception.fc = nn.Sequential(*classifier_layers)
        
        # 4. Disable aux classifier by overwriting with identity
        self.inception.aux_logits = False
        self.inception.AuxLogits = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Expected input shape:
            (batch_size, 3, H, W) — RGB images

        Returns:
            (batch_size, 1) logits
        """
        return self.inception(x)
