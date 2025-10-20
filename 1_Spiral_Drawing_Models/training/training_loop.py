from tqdm import tqdm 
import torch

# Training function loop for each epoch
# ---------------------------------------

def train_one_epoch(model:torch.nn.Module,
                    train_dataloader:torch.utils.data.DataLoader,
                    val_dataloader:torch.utils.data.DataLoader,

                    loss_fn:torch.nn.Module,
                    optim:torch.optim,
                    acc_fn,

                    scaler,
                    device):
    
    pass


# Vaildation function loop
# -------------------------
def validate(model: torch.nn.Module,
            val_dataloader: torch.utils.data.DataLoader,
            
            loss_fn: torch.nn.Module,
            scheduler:torch.optim.lr_scheduler,
            
            acc_fn,
            device):
    pass