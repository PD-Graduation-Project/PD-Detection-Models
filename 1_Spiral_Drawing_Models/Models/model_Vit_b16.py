from torchvision.models import vit_b_16
from torch import nn, inference_mode

def create_vit():
    """
    Creates a modified Vision Transformer (ViT-B/16) model for grayscale
    spiral drawing classification (e.g., Parkinson’s Disease detection).
    
    NOTE: This model only uses input size of '224' so make sure to change image_size in datasets

    The function performs the following steps:
        1. Loads a pretrained ViT-B/16 model with default ImageNet weights.
        2. Replaces the first convolutional layer to accept grayscale input (1 channel instead of 3).
        3. Averages the pretrained RGB weights to initialize the grayscale weights.
        4. Freezes all pretrained layers to retain learned features.
        5. Replaces the final classification head to output two classes (Healthy vs PD).

    Returns:
        nn.Module: A Vision Transformer model ready for fine-tuning on grayscale inputs.
    """
    
    # 1. load the pretrained ViT-B/16 model
    vit_model = vit_b_16(weights = "DEFAULT")
    
    # These next steps must be followed in this exact order
    # to insure the right params get freezed and not remove 
    # needed pretrained weights (of the input block)
    # -----------------------------------------------------
    
    # 2. get the pretraind input conv2d block
    old_conv = vit_model.conv_proj

    # 3. create the new conv2d input block, change the input to take 1 channel only (grayscale)
    new_conv = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))

    # 4. average the rgb weights across channels to form one grayscale channel
    with inference_mode():
        new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias[:] = old_conv.bias
        
    # 5. replace the original conv (input) block
    vit_model.conv_proj = new_conv

    # 6. freeze all pretrained params
    for param in vit_model.parameters():
        param.requires_grad = False

    # 7. change the output to 2 classes only (healthy, pd)
    # this only unfreezes this block
    vit_model.heads = nn.Sequential(
        nn.Linear(in_features=768, out_features=2)
    )
    
    return vit_model