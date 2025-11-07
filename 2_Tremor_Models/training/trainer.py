import torch
from .losses_binary import CombinedLoss, binary_metrics, compute_per_class_metrics
from .training_loop import train_one_epoch, validate
from .tensorboard_logger import log_metrics_to_tensorboard

from torch.utils.tensorboard import SummaryWriter
import os


def train(model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,

        load_pretrained: str = None,
        checkpoint_dir: str = "checkpoints/",
        model_name: str = "MODEL",

        run_name: str = "MODEL",
        Tboard: bool = True,

        epochs: int = 5,
        max_lr: float = 1e-4, 
        per_movement: bool = False):
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
        bce_weight=1.0,
        focal_weight=0.5,
        tversky_weight=0.5,
        
        healthy_weight=1.0,     
        parkinson_weight=1.5,
        
        focal_alpha=0.25,        
        focal_gamma=2.0,
        
        tversky_alpha=0.3,      
        tversky_beta=0.7,
    )

    # 1.2. optimizer (with weight decay)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=1e-4,
    )

    # 1.3. scheduler  (reduces LR when validation loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optim,
        mode='min',          # monitor val_loss
        factor=0.5,          # reduce LR by 50%
        patience=5,          # wait 2 epochs before reducing
        min_lr=1e-7,         # CHANGED: Set minimum LR to prevent too small values
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
            print(f"- val_accuracy={checkpoint['val_accuracy']:.4f}")
            print(f"- val_recall={checkpoint['val_recall']:.4f}")
            print(f"- val_precision={checkpoint['val_precision']:.4f}")
            print(f"- val_f1={checkpoint['val_f1']:.4f}")
        else:
            print(f"[WARNING] load_pretrained path was provided but does not exist: {load_pretrained}")

    # 5. full training loop
    # ----------------------
    for epoch in range(epochs):
        print(f"Training model: {model_name} epoch {epoch + 1} / {epochs}")
        print("-" * 35)

        # 6. train
        train_loss, train_overall, train_per_class = train_one_epoch(
            model= model,
            train_dataloader= train_dataloader,
            loss_fn= loss_fn,
            metric_fn= binary_metrics,
            compute_per_class_metrics= compute_per_class_metrics,
            optim= optim,
            scaler= scaler,
            device= device,
            per_movement= per_movement
        )

        # 7. validate
        val_loss, val_overall, val_per_class = validate(
            model= model,
            val_dataloader= val_dataloader,
            loss_fn= loss_fn,
            metric_fn= binary_metrics,
            compute_per_class_metrics= compute_per_class_metrics,
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
                train_per_class=train_per_class,
                val_per_class=val_per_class,
                current_lr=current_lr
            )

        # 9. Save the best model only
        # ------------------------------
        current_balanced_acc = val_per_class['balanced_accuracy']
        if current_balanced_acc > best_balanced_acc:
            best_balanced_acc = current_balanced_acc
            best_model_path = os.path.join(checkpoint_dir, f"{model_name}_BEST.pth")
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_overall': val_overall,
                'val_per_class': val_per_class,
                'balanced_accuracy': val_per_class['balanced_accuracy'],
                'macro_f1': val_per_class['macro_f1']
            }, best_model_path)
            
            print(f"[BEST] New best balanced_accuracy: {best_balanced_acc:.3f} (saved)\n")

        # 10. print epoch summary
        # -----------------------
        print(f"Epoch {epoch + 1} / {epochs} summary")
        print("-" * 35)
        
        print(f"  Loss: Train={train_loss:.4f}, Val={val_loss:.4f}")
        print(f"  Balanced Acc: Train={train_per_class['balanced_accuracy']:.3f}, Val={val_per_class['balanced_accuracy']:.3f}")
        print(f"  Macro F1: Train={train_per_class['macro_f1']:.3f}, Val={val_per_class['macro_f1']:.3f}")
        print(f"  Healthy F1: Train={train_per_class['healthy']['f1']:.3f}, Val={val_per_class['healthy']['f1']:.3f}")
        print(f"  PD F1: Train={train_per_class['parkinson']['f1']:.3f}, Val={val_per_class['parkinson']['f1']:.3f}")
        
        val_dist = val_per_class['prediction_dist']
        print(f"  Val Predictions: {val_dist['predicted_healthy']}H/{val_dist['predicted_pd']}PD "
                f"(Actual: {val_dist['actual_healthy']}H/{val_dist['actual_pd']}PD)")
        
        cm = val_per_class['confusion_matrix']
        print(f"  Confusion Matrix: TP={cm['TP']:.0f}, TN={cm['TN']:.0f}, FP={cm['FP']:.0f}, FN={cm['FN']:.0f}")
        
        print("=" * 35, "\n")

    # 11. close writer
    if Tboard:
        writer.close()