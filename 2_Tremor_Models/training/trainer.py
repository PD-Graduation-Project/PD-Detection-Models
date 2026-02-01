import torch
from .losses_binary import CombinedLoss, compute_metrics
from .training_loop import train_one_epoch, validate
from .tensorboard_logger import log_metrics_to_tensorboard

from torch.utils.tensorboard import SummaryWriter
import os

import yaml
from pathlib import Path

# Load config file
CONFIG_PATH = Path("..\config.yaml")

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()


def train(model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,

        load_pretrained: str = None,
        checkpoint_dir: str = "checkpoints/",
        model_name: str = "MODEL",

        run_name: str = "MODEL",
        Tboard: bool = True,

        epochs: int = config["trainer"]["epochs"],
        max_lr: float = config["trainer"]["lr"], 
        per_movement: bool = False,
        
        debug_mode:bool = True,):
    """
    Train a binary classification model with comprehensive per-class metrics tracking.
    
    Key improvements:
        - Per-class metrics (Healthy vs PD separately)
        - Prediction distribution tracking
        - Confusion matrix logging
        - Early stopping based on balanced metrics (not just loss)
        - Better learning rate and gradient clipping
        - Comprehensive TensorBoard logging

    Args:
        model (torch.nn.Module): Model to be trained.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        load_pretrained (str, optional): Path to a pretrained checkpoint. Defaults to None.
        checkpoint_dir (str, optional): Directory to save checkpoints. Defaults to "checkpoints/".
        model_name (str, optional): Base name for checkpoint files. Defaults to "MODEL".
        run_name (str, optional): TensorBoard run name (used for log directory). Defaults to "MODEL".
        Tboard (bool, optional): Whether to use SummaryWriter for TensorBoard logging. Defaults to True.
        epochs (int, optional): Number of training epochs. Defaults to 5.
        max_lr (float, optional): Maximum learning rate. Defaults to 1e-3.
        per_movement (bool): Whether the model has a movement embedding block or not.

    Notes:
        - Uses CombinedLoss (BCE + Focal + Tversky) optimized for class imbalance.
        - Optimizer: AdamW with weight decay and gradient clipping.
        - Scheduler: ReduceLROnPlateau based on validation loss.
        - Tracks: Overall metrics + Per-class metrics (Healthy/PD) + Confusion matrix.

    Returns:
        None
    """

    # 0. init device and add model to it
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 1. init loss, optimizer, scheduler, and scaler
    # ------------------------------------------------
    #  1.1. Combined loss function for binary classification with class imbalance
    # Adjusted for WeightedRandomSampler (batches are already balanced)
    loss_fn = CombinedLoss(
        bce_weight=config["losses"]["bce_weight"],
        focal_weight=config["losses"]["focal_weight"],
        tversky_weight=config["losses"]["tversky_weight"],
        
        healthy_weight=config["losses"]["healthy_weight"],     
        parkinson_weight=config["losses"]["parkinson_weight"],
        
        focal_alpha=config["losses"]["focal_alpha"],        
        focal_gamma=config["losses"]["focal_gamma"],
        
        tversky_alpha=config["losses"]["tversky_alpha"],      
        tversky_beta=config["losses"]["tversky_beta"],
    )

    # 1.2. optimizer (with weight decay)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=5e-5,
    )

    # 1.3. scheduler  (reduces LR when validation loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optim,
        mode='min',          # monitor val_loss
        factor=0.5,          # reduce LR by 50%
        patience=2,          # wait 2 epochs before reducing
        min_lr=1e-6,         # CHANGED: Set minimum LR to prevent too small values
        threshold= 0.01,     # Add threshold for meaningful improvement
    )

    # 1.4. scaler (to prevent underflow during mixed precision)
    scaler = torch.amp.GradScaler(device=device)

    # 2. init tensorboard writer
    # -----------------------------------------------
    if Tboard:
        writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    # 3. Best model tracking
    # ------------------------
    best_balanced_acc = 0.0

    # 4. load pretrained model if available
    # -------------------------------------
    if load_pretrained:
        if os.path.exists(load_pretrained):
            # load checkpoint
            checkpoint = torch.load(load_pretrained, map_location=device)
            # load model
            model.load_state_dict(checkpoint['model_state_dict'])
            # load optimizer
            optim.load_state_dict(checkpoint['optim_state_dict'])
            
            print(f"Loaded pretrained model:")
            print(f"- val_loss={checkpoint['val_loss']:.4f}")
            print(f"- balanced_accuracy={checkpoint['balanced_accuracy']:.4f}")
            print(f"- macro_f1={checkpoint['macro_f1']:.4f}")
        else:
            print(f"[WARNING] load_pretrained path was provided but does not exist: {load_pretrained}")

    # 5. full training loop
    # ----------------------
    for epoch in range(epochs):
        print(f"Training model: {model_name} epoch {epoch + 1} / {epochs}")
        print("-" * 35)

        # 6. train
        train_loss, train_overall = train_one_epoch(
            model= model,
            train_dataloader= train_dataloader,
            loss_fn= loss_fn,
            metric_fn= compute_metrics,
            optim= optim,
            scaler= scaler,
            device= device,
            per_movement= per_movement
        )

        # 7. validate
        val_loss, val_overall = validate(
            model= model,
            val_dataloader= val_dataloader,
            loss_fn= loss_fn,
            metric_fn= compute_metrics,
            device= device,
            per_movement= per_movement
        )
        
        # 7.1. step scheduler
        scheduler.step(val_loss)
        current_lr = optim.param_groups[0]['lr']
        

        # 8. log metrics to TensorBoard
        # ------------------------------------
        if Tboard:
            log_metrics_to_tensorboard(
                writer=writer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_overall=train_overall,
                val_overall=val_overall,
                current_lr=current_lr
            )

        # 9. Save the best model only
        # ------------------------------
        current_balanced_acc = val_overall['balanced_accuracy']
        if current_balanced_acc > best_balanced_acc:
            best_balanced_acc = current_balanced_acc
            best_model_path = os.path.join(checkpoint_dir, f"{model_name}_BEST.pth")
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_overall': val_overall,
                'balanced_accuracy': val_overall['balanced_accuracy'],
                'macro_f1': val_overall['macro_f1']
            }, best_model_path)
            
            print(f"[BEST] New best balanced_accuracy: {best_balanced_acc:.3f} (saved)\n")

        # 10. print epoch summary
        # -----------------------
        print(f"Epoch {epoch + 1} / {epochs} summary")
        print("-" * 35)
        
        print(f"  Loss: Train={train_loss:.4f}, Val={val_loss:.4f}")
        print(f"  Balanced Acc: Train={train_overall['balanced_accuracy']:.3f}, Val={val_overall['balanced_accuracy']:.3f}")
        
        # 11. Save per-movement stats to text file
        # -----------------------------------------
        if 'per_movement_counts' in val_overall:
            movement_stats = val_overall['per_movement_counts']
            movements_log_path = os.path.join(checkpoint_dir, f"{model_name}_per_movement_stats.txt")

            # Make sure checkpoint_dir exists
            os.makedirs(checkpoint_dir, exist_ok=True)

            with open(movements_log_path, "a") as f:  # append mode
                f.write(f"Epoch {epoch + 1}:\n")
                for m_id, stats in movement_stats.items():
                    f.write(f"  Movement {m_id}: Correct={stats['correct']}, Wrong={stats['wrong']}\n")
                f.write("\n")

        
        # 12. Extra details for debugging mode
        # -------------------------------------
        if debug_mode:
            print(f"  Macro F1: Train={train_overall['macro_f1']:.3f}, Val={val_overall['macro_f1']:.3f}")
            print(f"  Healthy F1: Train={train_overall['healthy']['f1']:.3f}, Val={val_overall['healthy']['f1']:.3f}")
            print(f"  PD F1: Train={train_overall['parkinson']['f1']:.3f}, Val={val_overall['parkinson']['f1']:.3f}")
            
            val_dist = val_overall['prediction_dist']
            print(f"  Val Predictions: {val_dist['predicted_healthy']}H/{val_dist['predicted_pd']}PD "
                    f"(Actual: {val_dist['actual_healthy']}H/{val_dist['actual_pd']}PD)")
            
            cm = val_overall['confusion_matrix']
            print(f"  Confusion Matrix: TP={cm['TP']:.0f}, TN={cm['TN']:.0f}, FP={cm['FP']:.0f}, FN={cm['FN']:.0f}")
        
        print("=" * 35, "\n")

    # 13. close writer
    if Tboard:
        writer.close()