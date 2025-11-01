import torch
from torch import nn
import torch.nn.functional as F

# Weighted Combined Losses class
# -------------------------------
class CombinedLoss(nn.Module):
    """
    Combined BinaryCrossEntropy + Focal + Tversky Loss optimized for detecting minority (Healthy) class 
    in a highly imbalanced dataset (Healthy=0, 22%; PD=1, 78%).

    Final Loss = w1*BCE + w2*Focal + w3*Tversky

    PARAMETER EFFECTS:
    ------------------
    - pos_weight: weight for minority (Healthy) class -> increases penalty for missing Healthy.
    - focal_alpha: balance between classes in Focal loss (↑ focuses more on Healthy)
    - focal_gamma: focuses more on hard-to-classify examples
    - tversky_alpha: weight for false negatives (↑ emphasizes Healthy recall)
    - tversky_beta: weight for false positives (↑ emphasizes precision)
    """
    
    def __init__(self,
                # loss weights
                bce_weight=1.0,
                focal_weight=1.2,
                tversky_weight=1.0,
                
                # weight for Healthy=0 (MINORITY)
                healthy_weight=3.5,       # num_PD / num_Healthy
                
                # focal params
                focal_alpha=0.15, # This focuses on majority (PD=1)
                focal_gamma=2.0,
                
                # tversky params
                tversky_alpha=0.8, # This focuses on minority (FN reduction)
                tversky_beta=0.2):
        super().__init__()
        
        # 1. init all params
        # -------------------
        
        # 1.1. Final combined loss weights
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        
        # 1.2. healthy_weight tensor (minority)
        self.register_buffer("healthy_weight", torch.tensor(healthy_weight))
        
        # 1.3. Focal loss params
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # 1.4. Tversky loss params
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

    def forward(self, pred, label):
        """
        pred: [B, 1] logits
        label: [B, 1] or [B] binary {0,1}
        """
        # 0. ensure correct shape
        pred = pred.view(-1, 1)
        label = label.view(-1, 1)
        
        probs = torch.sigmoid(pred)
        device = pred.device
        
        # 1. BCE with minority weighting
        weights = torch.where(label==0, # if healthy 
                            self.healthy_weight.to(device), # then apply weight
                            torch.ones_like(label, dtype=pred.dtype, device=device), # else make it one (the normal)
                            )
        bce = F.binary_cross_entropy_with_logits(probs, label.float(), weight=weights)
        
        # 2. Focal Loss
        # alpha: 0.25 for PD (1), 0.75 for Healthy (0)
        pt = torch.where(label == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.focal_gamma
        alpha_weight = torch.where(label == 1, self.focal_alpha, 1 - self.focal_alpha)
        bce_raw = F.binary_cross_entropy_with_logits(pred, label.float(), reduction='none')
        focal = (alpha_weight * focal_weight * bce_raw).mean()
        
        # 3. Tversky Loss
        # 3.1. True Positive, False Negative, False Positive
        TP = (probs * label).sum()               # correctly predicted PD
        FN = ((1 - probs) * label).sum()         # PD missed
        FP = (probs * (1 - label)).sum()         # predicted PD but actually Healthy
        
        # 3.2. Tversky index and loss (add epsilon to avoid division by zero)
        tversky_index = (TP + 1.0) / (TP + self.tversky_alpha * FN + self.tversky_beta * FP + 1.0)
        tversky = 1 - tversky_index
        
        # Combine
        total_loss = (self.bce_weight * bce +
                      self.focal_weight * focal +
                      self.tversky_weight * tversky)
        
        return total_loss


# Binary Metrics Function
# ------------------------
def binary_metrics(preds, labels, threshold=0.5):
    """
    Compute accuracy, recall, precision, and F1-score for binary classification.

    Args:
        preds (Tensor): raw logits from model, shape [B, 1] or [B]
        labels (Tensor): ground-truth labels, shape [B, 1] or [B]
        threshold (float): classification threshold on sigmoid output

    Returns:
        dict: {
            'accuracy': float,
            'recall': float,
            'precision': float,
            'f1': float
        }
    """
    probs = torch.sigmoid(preds).view(-1)
    labels = labels.view(-1)
    preds_bin = (probs >= threshold).float()
    
    tp = ((preds_bin == 1) & (labels == 1)).float().sum()
    fn = ((preds_bin == 0) & (labels == 1)).float().sum()
    fp = ((preds_bin == 1) & (labels == 0)).float().sum()
    
    accuracy = (preds_bin == labels).float().mean()
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        "accuracy": accuracy.item(),
        "recall": recall.item(),
        "precision": precision.item(),
        "f1": f1.item()
    }
