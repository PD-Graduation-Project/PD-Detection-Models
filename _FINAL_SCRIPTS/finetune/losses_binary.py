import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Optional

# Weighted Combined Losses class
# -------------------------------
class CombinedLoss(nn.Module):
    """
    Combined BinaryCrossEntropy + Focal + Tversky Loss for imbalanced binary classification.
    
    Designed for: 77% Parkinson (majority) vs 23% Healthy (minority)
    
    Final Loss = w1*BCE + w2*Focal + w3*Tversky
    """
    
    def __init__(self,
                # Loss component weights
                bce_weight=0.5,
                focal_weight=1.0,
                tversky_weight=0.5,
                label_smoothing = 0.03,
                
                # Class imbalance correction (for 77% PD, 23% Healthy)
                healthy_weight=1.5,    # Weight minority class higher
                parkinson_weight=1.0,   # Keep majority at 1.0
                
                # Focal loss params (focus on minority Healthy class)
                focal_alpha=0.23,  # Match minority proportion (focuses on Healthy)
                focal_gamma=2.0,   # Standard focusing strength
                
                # Tversky params (balanced F1-score)
                tversky_alpha=0.5,  # FN penalty (recall)
                tversky_beta=0.5):  # FP penalty (precision)
        super().__init__()
        
        # 1. Store all parameters
        # ------------------------
        self.label_smoothing = label_smoothing
        
        # 1.1. Final combined loss weights
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        
        # 1.2. Class weighting (corrected for minority class)
        self.register_buffer("healthy_weight", torch.tensor(healthy_weight))
        self.register_buffer("parkinson_weight", torch.tensor(parkinson_weight))
        
        # 1.3. Focal loss params
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # 1.4. Tversky loss params
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

    def forward(self, pred, label):
        """
        Args:
            pred: [B, 1] or [B] - raw logits from model
            label: [B, 1] or [B] - binary labels {0=Healthy, 1=Parkinson}
            
        Returns:
            total_loss: scalar tensor
        """
        # 0. Ensure correct shape
        pred = pred.view(-1, 1)
        label = label.view(-1, 1).float()
        
        device = pred.device
        
        # label smoothing
        smooth = self.label_smoothing
        label_smooth = label * (1 - smooth) + 0.5 * smooth
        
        # 1. Binary Cross-Entropy with class weighting
        # ----------------------------------------------
        # Apply higher weight to minority class (Healthy=0)
        weights = torch.where(
            label == 1,  # Parkinson (majority)
            self.parkinson_weight.to(device),
            self.healthy_weight.to(device)  # Healthy (minority) - higher weight
        )
        
        # FIXED: Use logits directly (no double sigmoid)
        bce = F.binary_cross_entropy_with_logits(pred, label_smooth, weight=weights)
        
        # 2. Focal Loss (focus on hard examples and minority class)
        # -----------------------------------------------------------
        # Convert logits to probabilities
        probs = torch.sigmoid(pred)
        
        # Calculate pt: probability of correct class
        pt = torch.where(label == 1, probs, 1 - probs)
        
        # Focal weight: (1-pt)^gamma focuses on hard examples
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # Alpha weight: balance between classes
        # focal_alpha < 0.5 focuses on minority (Healthy)
        alpha_weight = torch.where(label == 1, self.focal_alpha, 1 - self.focal_alpha)
        
        # Compute focal loss
        bce_raw = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        focal = (alpha_weight * focal_weight * bce_raw).mean()
        
        # 3. Tversky Loss (control precision-recall trade-off)
        # -----------------------------------------------------
        # Calculate True Positives, False Negatives, False Positives
        TP = (probs * label).sum()                    # Correctly predicted Parkinson
        FN = ((1 - probs) * label).sum()              # Missed Parkinson (↑ alpha to penalize)
        FP = (probs * (1 - label)).sum()              # False alarms (↑ beta to penalize)
        
        # Tversky index (closer to 1 = better)
        # Higher alpha penalizes FN more (improves recall)
        # Higher beta penalizes FP more (improves precision)
        tversky_index = (TP + 1.0) / (TP + self.tversky_alpha * FN + self.tversky_beta * FP + 1.0)
        tversky = 1 - tversky_index
        
        # 4. Combine all losses
        # ----------------------
        total_loss = (self.bce_weight * bce +
                      self.focal_weight * focal +
                      self.tversky_weight * tversky)
        
        return total_loss

# metrics computation
# ---------------------
def compute_metrics(preds, labels, movement_ids=None, threshold=0.5):
    """
    Compute overall and per-class metrics for binary classification.
    
    Args:
        preds: [N, 1] or [N] - raw logits
        labels: [N, 1] or [N] - binary labels {0=Healthy, 1=Parkinson}
        movement_ids: [N] integers 0..num_movements-1
        threshold: classification threshold
        
    Returns:
        dict with per-class metrics and confusion matrix
    """
    # 1. Convert to binary predictions
    probs = torch.sigmoid(preds).view(-1)
    labels = labels.view(-1)
    preds_bin = (probs >= threshold).float()
    
    # 2. Confusion matrix components
    # ==================================
    tp = ((preds_bin == 1) & (labels == 1)).float().sum()
    tn = ((preds_bin == 0) & (labels == 0)).float().sum()
    fp = ((preds_bin == 1) & (labels == 0)).float().sum()
    fn = ((preds_bin == 0) & (labels == 1)).float().sum()
    
    # 3. Overall metrics
    # ====================
    accuracy = (preds_bin == labels).float().mean()
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # 4. Per-class metrics
    # ==================================
    
    # 4.1. Healthy class (label=0, predicted as 0)
    # -----------------------------------------------
    healthy_recall = tn / (tn + fp + 1e-8)      # True Negative Rate (Specificity)
    healthy_precision = tn / (tn + fn + 1e-8)   # Negative Predictive Value
    healthy_f1 = 2 * (healthy_precision * healthy_recall) / (healthy_precision + healthy_recall + 1e-8)
    
    # 4.2. Parkinson class (label=1, predicted as 1)
    # -----------------------------------------------
    pd_recall = tp / (tp + fn + 1e-8)           # Sensitivity / True Positive Rate
    pd_precision = tp / (tp + fp + 1e-8)        # Positive Predictive Value
    pd_f1 = 2 * (pd_precision * pd_recall) / (pd_precision + pd_recall + 1e-8)
    
    # 5. Balanced metrics
    # =====================
    balanced_accuracy = (healthy_recall + pd_recall) / 2
    macro_f1 = (healthy_f1 + pd_f1) / 2
    
    # 6. Prediction distribution (to detect if model predicts all one class)
    # ============================
    total = len(labels)
    pred_healthy_count = (preds_bin == 0).sum().item()
    pred_pd_count = (preds_bin == 1).sum().item()
    
    actual_healthy_count = (labels == 0).sum().item()
    actual_pd_count = (labels == 1).sum().item()
    
    # 7. Per-movement metrics (optional)
    movement_stats = None
    if movement_ids is not None:
        movement_stats = movement_metrics(preds=preds,
                                        labels=labels,
                                        movement_ids=movement_ids)

    
    return {
        # 0. Confusion matrix
        # ---------------------
        'confusion_matrix': {
            'TP': tp.item(),
            'TN': tn.item(),
            'FP': fp.item(),
            'FN': fn.item()
        },
        
        # 1. Overall metrics
        # ---------------------
        'overall_metrics': {
        "accuracy": accuracy.item(),
        "recall": recall.item(),
        "precision": precision.item(),
        "f1": f1.item()
        },
        
        # 2. Healthy class (label=0)
        # ----------------------------
        'healthy': {
            'recall': healthy_recall.item(),        # How many Healthy we found
            'precision': healthy_precision.item(),  # How accurate our Healthy predictions are
            'f1': healthy_f1.item()
        },
        
        # 3. Parkinson class (label=1)
        # ------------------------------
        'parkinson': {
            'recall': pd_recall.item(),             # How many PD we found
            'precision': pd_precision.item(),       # How accurate our PD predictions are
            'f1': pd_f1.item()
        },
        
        # 4. Balanced metrics
        # --------------------
        'balanced_accuracy': balanced_accuracy.item(),
        'macro_f1': macro_f1.item(),
        
        # 5. Prediction distribution (detect if model is biased)
        # ----------------------------
        'prediction_dist': {
            'predicted_healthy': pred_healthy_count,
            'predicted_pd': pred_pd_count,
            'actual_healthy': actual_healthy_count,
            'actual_pd': actual_pd_count,
            'pred_healthy_ratio': pred_healthy_count / total,
            'pred_pd_ratio': pred_pd_count / total
        },
        
        # 6. Movements stats
        # -------------------
        'per_movement_counts': movement_stats  # None if movement_ids not provided
    }


def _binary_metrics(preds: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Compute binary metrics for a single-sample update."""
    probs = torch.sigmoid(preds).view(-1)
    labels = labels.view(-1)
    preds_bin = (probs >= threshold).float()

    tp = ((preds_bin == 1) & (labels == 1)).float().sum()
    fn = ((preds_bin == 0) & (labels == 1)).float().sum()
    fp = ((preds_bin == 1) & (labels == 0)).float().sum()

    accuracy = (preds_bin == labels).float().mean()
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {
        "accuracy": float(accuracy.item()),
        "recall": float(recall.item()),
        "precision": float(precision.item()),
        "f1": float(f1_score.item()),
    }