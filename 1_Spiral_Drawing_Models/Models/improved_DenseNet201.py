from torchvision.models import densenet201
from torch import nn, inference_mode

def create_improved_densenet(dropout_rate=0.5, 
                            hidden_units=[256, 64]):
    """
    Creates an improved DenseNet201 model for BINARY classification (single output).
    
    Key improvements inspired by MMNV2 paper:
    1. Multi-layer classifier head with dropout
    2. Global average pooling
    3. Deeper feature extraction before final classification
    4. SINGLE output neuron (for binary CE with logits)
    
    Args:
        dropout_rate: Dropout probability (default: 0.5)
        hidden_units: List of hidden layer sizes (default: [256, 64])
    
    Returns:
        Modified DenseNet201 model with binary output
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


    # 7. [NEW] build improved classifier
    classifier_layers = []
    
    # 7.1. build dense layers with dropout
    in_features=1920 # DenseNet201 output features
    
    for hidden_size in hidden_units:
        classifier_layers.extend([
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        in_features = hidden_size
        
    # 7.2. final layer: make the output to one class (for binary classification)
    classifier_layers.append( nn.Linear(in_features, 1) )
    
    # 8. replace final classifier with our new improved one
    densenet_model.classifier = nn.Sequential(*classifier_layers)

    return densenet_model
