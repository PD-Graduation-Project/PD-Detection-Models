from torchvision.models import vgg19_bn
from torch import nn, inference_mode

def create_vgg():
    """
    Creates a modified VGG19-BN model for grayscale input and binary classification.

    Steps:
    1. Load a pretrained VGG19-BN model (trained on ImageNet).
    2. Replace the first convolutional layer to accept 1-channel (grayscale) input.
    3. Initialize the new layer by averaging RGB weights.
    4. Freeze all pretrained layers to retain learned features.
    5. Replace the classifier head to output 2 classes (e.g., Healthy vs PD).

    Returns:
        nn.Module: Modified VGG19-BN model ready for training or inference.
    """

    # 1. Load the pretrained VGG19-BN model
    vgg_model = vgg19_bn(weights="DEFAULT")

    # 2. Get the original first convolutional layer
    old_conv = vgg_model.features[0]

    # 3. Create a new Conv2d layer with 1 input channel (for grayscale images)
    new_conv = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # 4. Initialize the grayscale layer by averaging the RGB weights
    with inference_mode():
        new_conv.weight = nn.Parameter(old_conv.weight.mean(dim=1, keepdim=True))
        if old_conv.bias is not None:
            new_conv.bias = old_conv.bias

    # 5. Replace the original conv layer with the new grayscale one
    vgg_model.features[0] = new_conv

    # 6. Freeze all pretrained layers
    for param in vgg_model.parameters():
        param.requires_grad = False

    # 7. Replace the classifier head to output 2 classes (unfreezes only this layer)
    vgg_model.classifier[-1] = nn.Linear(in_features=4096, out_features=2, bias=True)
    
    return vgg_model
