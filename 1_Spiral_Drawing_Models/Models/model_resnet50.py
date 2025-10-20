from torchvision.models import resnet50
from torch import nn, inference_mode

def create_resnet():
    """
    Creates a modified ResNet50 model for grayscale input and binary classification.

    Steps:
    1. Load a pretrained ResNet50 model (trained on ImageNet).
    2. Replace the first convolutional layer to accept 1-channel (grayscale) input.
    3. Initialize the new layer by averaging RGB weights.
    4. Freeze all pretrained layers to retain learned features.
    5. Replace the final fully connected layer to output 2 classes (e.g., Healthy vs PD).

    Returns:
        nn.Module: Modified ResNet50 model ready for training or inference.
    """

    # 1. Load pretrained ResNet50 model
    resnet_model = resnet50(weights="DEFAULT")

    # 2. Get the pretrained first convolutional layer
    old_conv = resnet_model.conv1

    # 3. Create a new Conv2d layer with 1 input channel (for grayscale images)
    new_conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # 4. Initialize the grayscale layer by averaging RGB weights
    with inference_mode():
        new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias[:] = old_conv.bias

    # 5. Replace the original conv layer with the new grayscale one
    resnet_model.conv1 = new_conv

    # 6. Freeze all pretrained layers
    for param in resnet_model.parameters():
        param.requires_grad = False

    # 7. Replace the final classification layer (unfreezes only this one)
    resnet_model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
    
    return resnet_model
