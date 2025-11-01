import torch
from .losses import CombinedLoss, binary_metrics
from .training_loop import train_one_epoch, validate

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
        max_lr: float = 1e-3,):
    """
    Train a multi-class classification model with TensorBoard logging and checkpoint saving.

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

    Notes:
        - Uses `CombinedLoss` (CrossEntropy + Focal + Tversky) for robust multi-class optimization.
        - Optimizer: AdamW with weight decay.
        - Scheduler: ReduceLROnPlateau.
        - Metrics: Accuracy, Recall, Precision, and F1 (macro-averaged).
        - Logs all metrics and LR to TensorBoard.
        - Saves checkpoints after each epoch with model and optimizer states.

    Returns:
        None
    """

    # 0. init device and add model to it
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 1. init loss, optimizer, scheduler, and scaler
    # ------------------------------------------------
    # 1.1. custom weighted loss function for 3-class setup
    class_weights = torch.tensor([1.85, 0.53, 1.28], device=device) # computed in readme file
    loss_fn = CombinedLoss(class_weights=class_weights)

    # 1.2. optimizer (with weight decay)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=1e-5,
    )

    # 1.3. scheduler 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optim,
        mode='min',          # monitor val_loss
        factor=0.5,          # reduce LR by 50%
        patience=2,          # wait 2 epochs before reducing
    )

    # 1.4. scaler (to prevent underflow during mixed precision)
    scaler = torch.amp.GradScaler(device=device)

    # 2. init tensorboard writer
    # -----------------------------------------------
    if Tboard:
        writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    # 3. load pretrained model if available
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
            print(f"- val_acc={checkpoint['val_acc']:.4f}")
            print(f"- val_recall={checkpoint['val_recall']:.4f}")
            print(f"- val_precision={checkpoint['val_precision']:.4f}")
            print(f"- val_f1={checkpoint['val_f1']:.4f}")
        else:
            print(f"[WARNING] load_pretrained path was provided but does not exist: {load_pretrained}")

    # 4. full training loop
    # ----------------------
    for epoch in range(epochs):
        print(f"Training model: {model_name} epoch {epoch + 1} / {epochs}")
        print("-" * 35)

        # 5. train
        train_loss, train_metrics = train_one_epoch(
            model,
            train_dataloader,
            loss_fn,
            binary_metrics,
            optim,
            scaler,
            device
        )

        # 6. validate
        val_loss, val_metrics = validate(
            model,
            val_dataloader,
            loss_fn,
            binary_metrics,
            device
        )
        scheduler.step(val_loss)


        if Tboard:
            # 7.1. log metrics to TensorBoard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)

            for m in ["accuracy", "recall", "precision", "f1"]:
                writer.add_scalar(f"{m}/train", train_metrics[m], epoch)
                writer.add_scalar(f"{m}/val", val_metrics[m], epoch)

            current_lr = optim.param_groups[0]['lr']
            writer.add_scalar("LearningRate", current_lr, epoch)
            writer.flush()

        # 8. save checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'val_loss': val_loss,
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }, os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch + 1}.pth"))

        print(f"Model: {model_name} saved.\n")

        # 9. print epoch summary
        print(f"Epoch {epoch + 1} / {epochs} summary")
        print("-" * 35)
        
        print(f"Train Loss = {train_loss:.3f} | "
            f"Acc: {train_metrics['accuracy']:.3f} | "
            f"Recall: {train_metrics['recall']:.3f} | "
            f"Precision: {train_metrics['precision']:.3f} | "
            f"F1: {train_metrics['f1']:.3f}")
        
        print(f"Val Loss = {val_loss:.3f} | "
            f"Acc: {val_metrics['accuracy']:.3f} | "
            f"Recall: {val_metrics['recall']:.3f} | "
            f"Precision: {val_metrics['precision']:.3f} | "
            f"F1: {val_metrics['f1']:.3f}")
        
        print("=" * 35, "\n")

    # 10. close writer
    if Tboard:
        writer.close()
