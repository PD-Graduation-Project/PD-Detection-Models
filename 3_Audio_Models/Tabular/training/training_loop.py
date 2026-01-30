from tqdm import tqdm 
import torch
from torch.nn.utils import clip_grad_norm_

# Training function loop for each epoch
# ---------------------------------------
def train_one_epoch(model:torch.nn.Module,
                    train_dataloader:torch.utils.data.DataLoader,

                    loss_fn:torch.nn.Module,
                    acc_fn, 
                    optim:torch.optim,
                    scheduler,
                    
                    scaler,
                    device):
    """
    Runs one epoch of training using mixed precision, weighted BCE loss, 
    and binary accuracy/precision/f1 tracking.
    """
    # 0. put model in train mode 
    model.train()
    
    # 1. init total losses and metrics
    total_losses = 0
    total_acc = 0
    total_recall = 0
    total_precision = 0
    total_f1 = 0
    
    # 1. loop through train_dataloader
    pbar = tqdm(
        iterable= train_dataloader,
        total= len(train_dataloader),
        desc="Training..."
    )
    
    for features, labels in pbar:
        # 2. move images and labels to device
        features = features.to(device)
        labels = labels.to(device).unsqueeze(1).float()  # shape [batch_size, 1] (float for BCE)
        
        # 3. enable auto mixed precision (AMP) for efficiency
        with torch.amp.autocast(device_type= device):
            # 4. forward pass
            logits = model(features)
            # if it's an Inception model, it may return a tuple
            if isinstance(logits, tuple):
                logits = logits[0]
                        
            # 5. calculate the losses, and the metrics (accuracy, recall, precision, f1)
            loss = loss_fn(logits, labels)
            acc, recall, precision, f1 = acc_fn(logits, labels)
            
            
        # 6. zero grad
        optim.zero_grad()
        
        # 7. scale loss and back propagate
        scaler.scale(loss).backward()
        
        # 8. gradient clipping to prevent exploading gradients
        scaler.unscale_(optim)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 9. step the opimizer, the scheduler, and update the scaler
        scaler.step(optim)
        scheduler.step()
        scaler.update()

        # 10. compute total loss and metrics
        total_losses += loss.item()
        total_acc += acc
        total_recall += recall
        total_precision += precision
        total_f1 += f1
        
        # 11. update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Accuracy': f'{acc:.4f}',
            'Recall': f'{recall:.4f}',
            'Precision': f'{precision:.4f}',
            'F1': f'{f1:.4f}'
        })
        
    # 12. return average losses and metrics
    avg_losses = total_losses / len(train_dataloader)
    avg_acc = total_acc/ len(train_dataloader)
    avg_recall = total_recall / len(train_dataloader)
    avg_precision = total_precision / len(train_dataloader)
    avg_f1 = total_f1 / len(train_dataloader)
    return avg_losses, avg_acc, avg_recall, avg_precision, avg_f1
            

# Vaildation function loop
# -------------------------
def validate(model: torch.nn.Module,
            val_dataloader: torch.utils.data.DataLoader,
            
            loss_fn: torch.nn.Module,
            
            acc_fn,
            device):
    """
    Runs validation after each training epoch using mixed precision and metrics tracking.
    """
    # 0. put model in eval mode 
    model.eval()
    
    # 1. init total losses and metrics
    total_losses = 0
    total_acc = 0
    total_recall = 0
    total_precision = 0
    total_f1 = 0
    
    # 2. loop through val_dataloader (in inference_mode)
    with torch.inference_mode():
        pbar = tqdm(
            iterable=val_dataloader,
            total=len(val_dataloader),
            desc="Testing..."
        )
        
        for features, labels in pbar:
            # 3. move images and labels to device
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1).float()  # shape [batch_size, 1] (float for BCE)
            
            # 4. enable auto mixed precision (AMP) for efficiency
            with torch.amp.autocast(device_type= device):
                # 5. forward pass
                logits = model(features)
                # if it's an Inception model, it may return a tuple
                if isinstance(logits, tuple):
                    logits = logits[0]
                                
                # 6. calculate the losses, and the metrics (accuracy, recall, precision, f1)
                loss = loss_fn(logits, labels)
                acc, recall, precision, f1 = acc_fn(logits, labels)
                
            # 7. compute total loss and metrics
            total_losses += loss.item()
            total_acc += acc
            total_recall += recall
            total_precision += precision
            total_f1 += f1
            
            # 8. update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Accuracy': f'{acc:.4f}',
                'Recall': f'{recall:.4f}',
                'Precision': f'{precision:.4f}',
                'F1': f'{f1:.4f}'
            })
            
        # 9. return average losses and metrics
        avg_losses = total_losses / len(val_dataloader)
        avg_acc = total_acc/ len(val_dataloader)
        avg_recall = total_recall / len(val_dataloader)
        avg_precision = total_precision / len(val_dataloader)
        avg_f1 = total_f1 / len(val_dataloader)
        return avg_losses, avg_acc, avg_recall, avg_precision, avg_f1