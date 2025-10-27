import torch
from torch import nn

class TremorNetGRU(nn.Module):
    def __init__(self,
                num_classes: int = 3,
                hidden_size: int = 128,
                ):
        super().__init__()
        
        