from torchvision.models import densenet201
from torch import nn, inference_mode

def create_densenet():
    """
    Creates a modified pretrained DenseNet201 model for grayscale input and binary classification (healthy vs PD).

    Steps:
    ------
    1. Load pretrained DenseNet201 model.
    2. Replace the first convolutional layer to accept 1-channel (grayscale) input.
    3. Copy pretrained RGB weights by averaging across channels.
    4. Freeze all pretrained parameters.
    5. Replace the classifier layer for 2-class output.
    """
    # 1. load pretrained DenseNet201 model
    densenet_model = densenet201(weights="DEFAULT")

    # 2. get the pretrained weights of the original conv2d block (input)
    old_conv = densenet_model.features[0]

    # 3. create new conv2d layer for 1-channel (grayscale) input
    new_conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # 4. average RGB weights to form grayscale weights
    with inference_mode():
        new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias[:] = old_conv.bias

    # 5. replace the input conv layer
    densenet_model.features[0] = new_conv

    # 6. freeze model parameters
    for param in densenet_model.parameters():
        param.requires_grad = False

    # 7. modify final classifier for binary classification
    densenet_model.classifier = nn.Linear(in_features=1920, out_features=2, bias=True)

    return densenet_model
