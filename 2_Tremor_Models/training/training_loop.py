from tqdm import tqdm 
import torch
from torch.nn.utils import clip_grad_norm_

# Training function loop for each epoch
# ---------------------------------------
def train_one_epoch(model: torch.nn.Module,
                    train_dataloader: torch.utils.data.DataLoader,
                    
                    loss_fn: torch.nn.Module,
                    metric_fn, 
                    optim: torch.optim,
                    
                    scaler,
                    device,
                    per_movement):
    """
    Runs one epoch of multi-class training using mixed precision and metrics tracking.

    Uses macro-averaged accuracy, recall, precision, and F1-score.
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
        iterable=train_dataloader,
        total=len(train_dataloader),
        desc="Training..."
    )
    
    for batch in pbar:
        # 2. unpack and move data to device
        if not per_movement:
            signals, wrists, movements, labels = [b.to(device) for b in batch]  # (B, T, 6), (B,), (B,), (B,)
        else:
            signals, wrists, labels = [b.to(device) for b in batch]  # (B, T, 6), (B,), (B,)
        
        # 3. enable auto mixed precision (AMP) for efficiency
        with torch.amp.autocast(device_type=device):
            
            # 4. forward pass (model takes signal, wrist, and movement)
            if not per_movement:
                logits = model(signals, wrists, movements)
            else:
                logits = model(signals, wrists)
                        
            # 5. calculate the losses and metrics
            loss = loss_fn(logits, labels)
            metrics = metric_fn(logits, labels)
            
        # 6. zero grad
        optim.zero_grad()
        
        # 7. scale loss and back propagate
        scaler.scale(loss).backward()
        
        # 8. gradient clipping to prevent exploding gradients
        scaler.unscale_(optim)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 9. step the optimizer, scheduler, and update scaler
        scaler.step(optim)
        scaler.update()

        # 10. compute total loss and metrics
        total_losses += loss.item()
        total_acc += metrics['accuracy']
        total_recall += metrics['recall']
        total_precision += metrics['precision']
        total_f1 += metrics['f1']
        
        # 11. update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{metrics["accuracy"]:.4f}',
            'Rec': f'{metrics["recall"]:.4f}',
            'Prec': f'{metrics["precision"]:.4f}',
            'F1': f'{metrics["f1"]:.4f}'
        })
        
    # 12. return average losses and metrics as dictionary
    avg_loss = total_losses / len(train_dataloader)

    metrics_dict = {
        "accuracy": total_acc / len(train_dataloader),
        "recall": total_recall / len(train_dataloader),
        "precision": total_precision / len(train_dataloader),
        "f1": total_f1 / len(train_dataloader),
    }

    return avg_loss, metrics_dict
            

# Validation function loop
# -------------------------
def validate(model: torch.nn.Module,
            val_dataloader: torch.utils.data.DataLoader,
            loss_fn: torch.nn.Module,
            metric_fn,
            device,
            per_movement):
    """
    Runs multi-class validation using mixed precision and macro-averaged metrics.

    Macro averaging ensures each class contributes equally to recall, precision, and F1,
    providing a more balanced evaluation for imbalanced datasets.
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
        
        for batch in pbar:
            # 3. unpack and move data to device
            if not per_movement:
                signals, wrists, movements, labels = [b.to(device) for b in batch]
            else:
                signals, wrists, labels = [b.to(device) for b in batch]
                
            # 4. enable auto mixed precision (AMP)
            with torch.amp.autocast(device_type=device):
                # 5. forward pass
                if not per_movement:
                    logits = model(signals, wrists, movements)
                else:
                    logits = model(signals, wrists)
                                
                # 6. calculate losses and metrics
                loss = loss_fn(logits, labels)
                metrics = metric_fn(logits, labels)
                
            # 7. compute total loss and metrics
            total_losses += loss.item()
            total_acc += metrics['accuracy']
            total_recall += metrics['recall']
            total_precision += metrics['precision']
            total_f1 += metrics['f1']
            
            # 8. update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{metrics["accuracy"]:.4f}',
                'Rec': f'{metrics["recall"]:.4f}',
                'Prec': f'{metrics["precision"]:.4f}',
                'F1': f'{metrics["f1"]:.4f}'
            })
            
        # 9. return average loss and metrics as dictionary
        avg_loss = total_losses / len(val_dataloader)

        metrics_dict = {
            "accuracy": total_acc / len(val_dataloader),
            "recall": total_recall / len(val_dataloader),
            "precision": total_precision / len(val_dataloader),
            "f1": total_f1 / len(val_dataloader),
        }

        return avg_loss, metrics_dict

