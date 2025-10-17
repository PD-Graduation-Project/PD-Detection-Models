from torch.utils.data import Dataset
import albumentations as A
from PIL import Image

# Dataset class
class DrawingDataset(Dataset):
    def __init__(self,
                healthy_dir:str,
                pd_dir:str,
                img_size:int = 512,
                transforms:list = None):
        super().__init__()
        
        