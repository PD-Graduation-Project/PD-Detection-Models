from torchvision.models import inception_v3
from torch import nn, inference_mode

def create_inception():
    """
    Creates a modified pretrained InceptionV3 model for grayscale input and binary classification (healthy vs PD).

    Steps:
    ------
    1. Load pretrained InceptionV3 model.
    2. Replace the first convolutional layer to accept 1-channel (grayscale) input.
    3. Copy pretrained RGB weights by averaging across channels.
    4. Disable Inception's internal RGB input transform.
    5. Freeze all pretrained parameters.
    6. Replace the classifier (fc) layer for 2-class output.
    """
    # 1. load pretrained model
    inception_model = inception_v3(weights="DEFAULT")

    # 2. get the pretrained weights of the original conv2d block (input)
    old_conv = inception_model.Conv2d_1a_3x3.conv

    # 3. create new conv2d layer for 1-channel (grayscale) input
    new_conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

    # 4. average RGB weights to create grayscale weights
    with inference_mode():
        new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias[:] = old_conv.bias

    # 5. replace the input conv layer
    inception_model.Conv2d_1a_3x3.conv = new_conv

    # 6. disable RGB normalization step (Inception expects 3 channels by default)
    inception_model._transform_input = lambda x: x

    # 7. freeze model parameters
    for param in inception_model.parameters():
        param.requires_grad = False

    # 8. modify final classifier for binary classification
    inception_model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)

    return inception_model
