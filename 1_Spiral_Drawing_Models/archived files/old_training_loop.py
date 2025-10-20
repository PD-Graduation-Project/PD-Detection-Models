from tqdm import tqdm 
import torch

# Training function loop for each epoch
# ---------------------------------------
def train_one_epoch(model:torch.nn.Module,
                    train_dataloader:torch.utils.data.DataLoader,

                    loss_fn:torch.nn.Module,
                    optim:torch.optim,
                    acc_fn, # now this is a function: binary_accuracy(preds, labels)

                    scaler,
                    device):
    """
    Runs one epoch of training using mixed precision, weighted BCE loss, 
    and binary accuracy tracking.
    """
    # 0. put model in train mode 
    model.train()
    
    # 1. init total losses and accuracy
    total_losses = 0
    total_acc = 0
    
    # 1. loop through train_dataloader
    pbar = tqdm(
        iterable= train_dataloader,
        total= len(train_dataloader),
        desc="Training..."
    )
    
    for batch in pbar:
        # 2. move images and labels to device
        imgs = batch['image'].to(device)
        labels = batch['label'].to(device).unsqueeze(1).float()  # shape [batch_size, 1] (float for BCE)
        
        # 3. enable auto mixed precision (AMP) for efficiency
        with torch.amp.autocast(device_type= device):
            # 4. forward pass
            logits = model(imgs)
            # if it's an Inception model, it may return a tuple
            if isinstance(logits, tuple):
                logits = logits[0]
                        
            # 5. calculate the losses, and the accuracy
            loss = loss_fn(logits, labels)
            acc = acc_fn(logits, labels)
            
            
        # 6. zero grad
        optim.zero_grad()
        
        # 7. scale loss and back propagate
        scaler.scale(loss).backward()
        
        # 8. step the opimizer and update the scaler
        scaler.step(optim)
        scaler.update()

        # 9. compute total loss and accuracy
        total_losses += loss.item()
        total_acc += acc.item()
        
        # 10. update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Accuracy': f'{acc.item():.4f}'
        })
        
    # 11. return average losses and accuracy
    avg_losses = total_losses / len(train_dataloader)
    avg_acc = total_acc/ len(train_dataloader)
    return avg_losses, avg_acc
            

# Vaildation function loop
# -------------------------
def validate(model: torch.nn.Module,
            val_dataloader: torch.utils.data.DataLoader,
            
            loss_fn: torch.nn.Module,
            
            acc_fn,
            device):
    """
    Runs validation after each training epoch using mixed precision and accuracy tracking.
    """
    # 0. put model in eval mode 
    model.eval()
    
    # 1. init total losses and accuracy
    total_losses = 0
    total_acc = 0
    
    # 2. loop through val_dataloader (in inference_mode)
    with torch.inference_mode():
        pbar = tqdm(
            iterable=val_dataloader,
            total=len(val_dataloader),
            desc="Testing..."
        )
        
        for batch in pbar:
            # 3. move images and labels to device
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device).unsqueeze(1).float()  # shape [batch_size, 1] (float for BCE)
            
            # 4. enable auto mixed precision (AMP) for efficiency
            with torch.amp.autocast(device_type= device):
                # 5. forward pass
                logits = model(imgs)
                # if it's an Inception model, it may return a tuple
                if isinstance(logits, tuple):
                    logits = logits[0]
                                
                # 6. calculate the losses, and the accuracy
                loss = loss_fn(logits, labels)
                acc = acc_fn(logits, labels)
                
            # 7. compute total loss and accuracy
            total_losses += loss.item()
            total_acc += acc.item()
            
            # 8. update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Accuracy': f'{acc.item():.4f}'
            })
            
        # 9. return average losses and accuracy
        avg_losses = total_losses / len(val_dataloader)
        avg_acc = total_acc/ len(val_dataloader)
        return avg_losses, avg_acc