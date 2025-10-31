import torch
from .losses import CombinedLoss, binary_metrics
from .training_loop import train_one_epoch, validate

from torch.utils.tensorboard import SummaryWriter
import os

def train(model:torch.nn.Module,
        train_dataloader:torch.utils.data.DataLoader,
        val_dataloader:torch.utils.data.DataLoader,
        
        load_pretrained:str = None,
        checkpoint_dir:str = "checkpoints/",
        model_name:str = "MODEL",
        
        run_name:str = "MODEL",
        Tboard:bool = True,
        
        epochs:int = 5,
        max_lr:float = 1e-3,):
    
    """
    Train a binary classification model with TensorBoard logging and checkpoint saving.

    Args:
        model (torch.nn.Module): Model to be trained.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        load_pretrained (str, optional): Path to a pretrained checkpoint. Defaults to None.
        checkpoint_dir (str, optional): Directory to save checkpoints. Defaults to "checkpoints/".
        model_name (str, optional): Base name for checkpoint files. Defaults to "MODEL".
        run_name (str, optional): TensorBoard run name (used for log directory). Defaults to "MODEL".
        Tboard (bool, optional): wheater to use SummaryWriter or not
        epochs (int, optional): Number of training epochs. Defaults to 5.
        max_lr (float, optional): Initial learning rate. Defaults to 1e-3.

    Notes:
        - Uses `WeightedBCE` loss to emphasize false negatives.
        - Optimizer: AdamW with weight decay.
        - Scheduler: replaced `ReduceLROnPlateau` with `OneCycleLR` (More aggressive).
        - Accuracy metric: simple binary accuracy function.
        - Logs train/val loss, accuracy, and LR to TensorBoard.
        - Saves a checkpoint after each epoch with model and optimizer states.

    Returns:
        None
    """
    
    # 0. init device and add model to it
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # 1. init loss, optimizer, scheduler, and scaler
    # ------------------------------------------------
    
    # 1.1. custome weighted loss function
    loss_fn = CombinedLoss()
    
    # 1.2. optimizer (with weight decay)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr = max_lr,
        weight_decay= 1e-5,
    )
    
    # 1.3. scheduler 
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer= optim,
        max_lr= max_lr,
        epochs=epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.3,            # warm up for 30% of training (default)
        div_factor=25,            # Start LR = max_lr/25 (more conservative start) (default)
        final_div_factor=1e4      # End LR = max_lr/10000 (fine-tune at end) (default)
    )
    
    # 1.4. scaler (to prevent underflow)
    scaler = torch.amp.GradScaler(device = device)
    
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
            #load optimizer
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
        print(f"Training model:{model_name} epoch no.{epoch+1} / {epochs}")
        print("-"*35)
        
        # 5. train
        train_loss, train_acc, train_recall, train_precision, train_f1  = train_one_epoch(
            model,
            train_dataloader,
            
            loss_fn,
            binary_metrics,
            optim,
            scheduler,
            
            scaler,
            device
        )
        
        # 6. validate
        val_loss, val_acc, val_recall, val_precision, val_f1 = validate(
            model,
            val_dataloader,
            
            loss_fn,
            
            binary_metrics,
            device
        )
        
        
        if Tboard:
            # 7.1 log metrics to tensorboard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            
            writer.add_scalar("Recall/train", train_recall, epoch)
            writer.add_scalar("Recall/val", val_recall, epoch)
            
            writer.add_scalar("Precision/train", train_precision, epoch)
            writer.add_scalar("Precision/val", val_precision, epoch)
            
            writer.add_scalar("F1/train", train_f1, epoch)
            writer.add_scalar("F1/val", val_f1, epoch)
            writer.flush()  # ensure logs are written immediately
            
            # 7.2. log learning rate (useful with ReduceLROnPlateau)
            current_lr = optim.param_groups[0]['lr']
            writer.add_scalar("LearningRate", current_lr, epoch)
        
        
        # 8. save checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            
            'val_loss': val_loss,
            'val_acc':val_acc,
            'val_recall':val_recall,
            'val_precision': val_precision,
            'val_f1': val_f1,
        }, os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch+1}.pth"))

        print(f"Model: {model_name} saved.\n")
        
        # 9. print epoch summary
        print(f"Epoch no.{epoch+1} / {epochs} summary")
        print("-"*35)
        print(f"Average train losses = {train_loss:.3f} | Train Acc: {train_acc:.3f} | Train Recall: {train_recall:.3f} | Train Precision: {train_precision:.3f} | Train F1: {train_f1:.3f}")
        print(f"Average validation losses = {val_loss:.3f} | Val Acc:   {val_acc:.3f} | Val Recall:   {val_recall:.3f} | Val Precision: {val_precision:.3f} | Val F1: {val_f1:.3f}")
        print("="*35, "\n")
        
    # 10. close writer
    if Tboard:
        writer.close()