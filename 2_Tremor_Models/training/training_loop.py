from tqdm import tqdm 
import torch
from torch.nn.utils import clip_grad_norm_

# Training function loop for each epoch
# ---------------------------------------
def train_one_epoch(model: torch.nn.Module,
                    train_dataloader: torch.utils.data.DataLoader,
                    
                    loss_fn: torch.nn.Module,
                    metric_fn, 
                    compute_per_class_metrics,
                    optim: torch.optim,
                    
                    scaler,
                    device,
                    per_movement):
    """
    Runs one epoch of binary training with per-class metrics tracking.
    
    Returns both overall metrics and per-class breakdown for Healthy and PD.
    """
    # 0. put model in train mode 
    model.train()
    
    # 1. init total losses and metrics
    total_losses = 0
    
    # 1.1. Store all predictions and labels for per-class metrics
    all_preds = []
    all_labels = []
    
    # 2. loop through train_dataloader
    pbar = tqdm(
        iterable=train_dataloader,
        total=len(train_dataloader),
        desc="Training..."
    )
    
    for batch in pbar:
        # 3. unpack and move data to device
        signals, handedness, movements, labels = [b.to(device) for b in batch]  # (B, 2, T, 6), (B,), (B,), (B,)
        
        # 4. enable auto mixed precision (AMP) for efficiency
        with torch.amp.autocast(device_type=device):
            
            # 5. forward pass (model takes signals, handedness, and movement)
            if not per_movement:
                logits = model(signals, handedness, movements)
            else:
                logits = model(signals, handedness)
                        
            # 6. calculate the loss
            loss = loss_fn(logits, labels)
            
        # 7. zero grad
        optim.zero_grad()
        
        # 8. scale loss and back propagate
        scaler.scale(loss).backward()
        
        # 9. gradient clipping to prevent exploding gradients
        scaler.unscale_(optim)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 10. step the optimizer, scheduler, and update scaler
        scaler.step(optim)
        scaler.update()

        # 11. store predictions and labels for metrics
        all_preds.append(logits.detach())
        all_labels.append(labels.detach())
        
        # 12. compute total loss
        total_losses += loss.item()
        
        # 13. update progress bar with batch loss
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
    # 14. Compute overall metrics on entire epoch
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    overall_metrics = metric_fn(all_preds, all_labels)
    
    # 15. Compute per-class metrics
    per_class_metrics = compute_per_class_metrics(all_preds, all_labels)
    
    # 16. return average loss, overall metrics, and per-class metrics
    avg_loss = total_losses / len(train_dataloader)

    return avg_loss, overall_metrics, per_class_metrics
            

# Validation function loop
# -------------------------
def validate(model: torch.nn.Module,
            val_dataloader: torch.utils.data.DataLoader,
            loss_fn: torch.nn.Module,
            metric_fn,
            compute_per_class_metrics,
            device,
            per_movement):
    """
    Runs binary validation with per-class metrics tracking.
    
    Returns both overall metrics and per-class breakdown for Healthy and PD.
    """
    # 0. put model in eval mode 
    model.eval()
    
    # 1. init total losses
    total_losses = 0
    
    # Store all predictions and labels for per-class metrics
    all_preds = []
    all_labels = []
    
    # 2. loop through val_dataloader (in inference_mode)
    with torch.inference_mode():
        pbar = tqdm(
            iterable=val_dataloader,
            total=len(val_dataloader),
            desc="Validating..."
        )
        
        for batch in pbar:
            # 3. unpack and move data to device
            signals, handedness, movements, labels = [b.to(device) for b in batch]
                
            # 4. enable auto mixed precision (AMP)
            with torch.amp.autocast(device_type=device):
                # 5. forward pass
                if not per_movement:
                    logits = model(signals, handedness, movements)
                else:
                    logits = model(signals, handedness)
                                
                # 6. calculate loss
                loss = loss_fn(logits, labels)
                
            # 7. store predictions and labels
            all_preds.append(logits.detach())
            all_labels.append(labels.detach())
            
            # 8. compute total loss
            total_losses += loss.item()
            
            # 9. update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
    # 10. Compute overall metrics on entire epoch
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    overall_metrics = metric_fn(all_preds, all_labels)
    
    # 11. Compute per-class metrics
    per_class_metrics = compute_per_class_metrics(all_preds, all_labels)
    
    # 12. return average loss, overall metrics, and per-class metrics
    avg_loss = total_losses / len(val_dataloader)

    return avg_loss, overall_metrics, per_class_metrics

