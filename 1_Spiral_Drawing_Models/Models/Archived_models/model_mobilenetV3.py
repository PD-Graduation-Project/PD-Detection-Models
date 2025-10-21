from torchvision.models import mobilenet_v3_large
from torch import nn, inference_mode

def create_mobilenetv3():
    """
    Creates a modified pretrained MobileNetV3-Large model for grayscale input and binary classification (healthy vs PD).

    Steps:
    ------
    1. Load pretrained MobileNetV3-Large model.
    2. Replace the first convolutional layer to accept 1-channel (grayscale) input.
    3. Copy pretrained RGB weights by averaging across channels.
    4. Freeze all pretrained parameters.
    5. Replace the classifier layer for binary output.
    """
    # 1. load pretrained MobileNetV3-Large model
    mobilenet_model = mobilenet_v3_large(weights="DEFAULT")

    # 2. get the pretrained Conv2d inside Conv2dNormActivation
    old_conv = mobilenet_model.features[0][0]  # Conv2d inside Conv2dNormActivation

    # 3. create new Conv2d layer for 1-channel (grayscale) input
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    )

    # 4. average RGB weights to form grayscale weights
    with inference_mode():
        new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias[:] = old_conv.bias

    # 5. replace the input conv layer
    mobilenet_model.features[0][0] = new_conv

    # 6. freeze all parameters
    for param in mobilenet_model.parameters():
        param.requires_grad = False

    # 7. modify final classifier for binary output
    mobilenet_model.classifier[3] = nn.Linear(in_features=1280, out_features=1, bias=True)

    return mobilenet_model
