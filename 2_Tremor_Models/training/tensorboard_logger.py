from torch.utils.tensorboard import SummaryWriter


def log_metrics_to_tensorboard(
        writer: SummaryWriter,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_overall: dict,
        val_overall: dict,
        train_per_class: dict,
        val_per_class: dict,
        current_lr: float):
    """
    Logs all training metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter instance
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        train_overall: Overall metrics dict from binary_metrics()
        val_overall: Overall validation metrics dict
        train_per_class: Per-class metrics dict from compute_per_class_metrics()
        val_per_class: Per-class validation metrics dict
        current_lr: Current learning rate
    """
    
    # 1. Loss
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    
    # 2. Overall metrics
    for metric_name in ["accuracy", "recall", "precision", "f1"]:
        writer.add_scalar(f"Overall/{metric_name}/train", train_overall[metric_name], epoch)
        writer.add_scalar(f"Overall/{metric_name}/val", val_overall[metric_name], epoch)
        
    # 3. Per-class metrics
    for class_name in ["healthy", "parkinson"]:
        for metric in ["recall", "precision", "f1"]:
            writer.add_scalar(
                f"PerClass/{class_name}_{metric}/train",
                train_per_class[class_name][metric],
                epoch
            )
            writer.add_scalar(
                f"PerClass/{class_name}_{metric}/val",
                val_per_class[class_name][metric],
                epoch
            )
            
    # 4. Balanced metrics
    writer.add_scalar("Balanced/accuracy/train", train_per_class['balanced_accuracy'], epoch)
    writer.add_scalar("Balanced/accuracy/val", val_per_class['balanced_accuracy'], epoch)
    writer.add_scalar("Balanced/macro_f1/train", train_per_class['macro_f1'], epoch)
    writer.add_scalar("Balanced/macro_f1/val", val_per_class['macro_f1'], epoch)
    
    # 5. Prediction distribution (detect bias)
    writer.add_scalar(
        "PredDist/train_healthy_ratio",
        train_per_class['prediction_dist']['pred_healthy_ratio'],
        epoch
    )
    writer.add_scalar(
        "PredDist/train_pd_ratio",
        train_per_class['prediction_dist']['pred_pd_ratio'],
        epoch
    )
    writer.add_scalar(
        "PredDist/val_healthy_ratio",
        val_per_class['prediction_dist']['pred_healthy_ratio'],
        epoch
    )
    writer.add_scalar(
        "PredDist/val_pd_ratio",
        val_per_class['prediction_dist']['pred_pd_ratio'],
        epoch
    )
    
    # 6. Confusion matrix as scalars
    for key, value in val_per_class['confusion_matrix'].items():
        writer.add_scalar(f"ConfusionMatrix/{key}", value, epoch)
    
    # 7. Learning rate
    writer.add_scalar("LearningRate", current_lr, epoch)
    
    # Flush to disk
    writer.flush()