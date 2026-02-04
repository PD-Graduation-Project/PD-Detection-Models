import torch
from torch import nn
import torch.nn.functional as F

# Weighted Combined Losses class
# -------------------------------
class CombinedLoss(nn.Module):
    """
    Combined BCE + Focal + Tversky Loss optimized for extreme recall (minimize false negatives).
    
    Final Loss = w1*BCE + w2*Focal + w3*Tversky
    
    HYPERPARAMETER EFFECTS:
    -----------------------
    pos_weight (default=4.0):
        ↑ Higher = MORE penalty on FN, model predicts PD more often
        ↓ Lower = LESS penalty on FN, model predicts Healthy more often
    
    focal_alpha (default=0.85):
        ↑ Higher = focus MORE on positive class (PD cases)
        ↓ Lower = focus MORE on negative class (Healthy cases)
    
    focal_gamma (default=3.0):
        ↑ Higher = focus MORE on hard-to-classify examples
        ↓ Lower = treat all examples more equally
        (0 = regular BCE, 2-3 = typical, 5 = extreme focus)
    
    tversky_alpha (default=0.85):
        ↑ Higher = penalize False Negatives MORE (miss PD cases)
        ↓ Lower = penalize False Negatives LESS
    
    tversky_beta (default=0.15):
        ↑ Higher = penalize False Positives MORE (false alarms)
        ↓ Lower = penalize False Positives LESS
        
    Note: tversky_alpha + tversky_beta should ≈ 1.0
    
    Component weights (bce_weight, focal_weight, tversky_weight):
        ↑ Higher = that loss component has MORE influence
        ↓ Lower = that loss component has LESS influence
    """
    def __init__(self,
                # losses weights (final equation)
                bce_weight=1.0,
                focal_weight=0.5,
                tversky_weight=0.5,
                
                # positive class (PD) weight
                pos_weight=1.5,
                
                # focal params
                focal_alpha=0.6,
                focal_gamma=1.5,
                
                # tversky params
                tversky_alpha=0.60,
                tversky_beta=0.40):
        super().__init__()
        
        # 1. init all params
        # -------------------
        
        # 1.1. Final combined loss weights
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        
        # 1.2. `pos_weight` applies a multiplier only to the positive class (making the model care more about it)
        self.pos_weight_tensor = torch.tensor([pos_weight])
        
        # 1.3. Focal loss params
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # 1.4. Tversky loss params
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        
    # 2. Forward function
    # ----------------------
    def forward(self, pred, label):
        # 0. ensure both predictions and labels have the SAME shape (N, 1) (can use flatten())
        pred = pred.view(-1, 1)
        label = label.view(-1, 1)
        
        # 1. Weighted BCE
        pos_weight = self.pos_weight_tensor.to(pred.device)
        bce = F.binary_cross_entropy_with_logits(pred, label, pos_weight=pos_weight)
        
        # 2. Focal Loss
        probs = torch.sigmoid(pred)
        pt = torch.where(label == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.focal_gamma
        alpha_weight = torch.where(label == 1, self.focal_alpha, 1 - self.focal_alpha)
        bce_raw = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        focal = (alpha_weight * focal_weight * bce_raw).mean()
        
        # 3. Tversky Loss
        TP = (probs * label).sum()
        FN = ((1 - probs) * label).sum()
        FP = (probs * (1 - label)).sum()
        tversky_index = (TP + 1.0) / (TP + self.tversky_alpha * FN + self.tversky_beta * FP + 1.0)
        tversky = 1 - tversky_index
        
        # Combine
        total_loss = (self.bce_weight * bce + 
                      self.focal_weight * focal + 
                      self.tversky_weight * tversky)
        
        return total_loss
    
    
    
# Simple binary metrics function
# --------------------------------
def binary_metrics(preds, labels, threshold=0.5):
    """
    Compute binary accuracy, recall, percision, and F1-score
    
    Returns:
        accuracy (float): Overall accuracy
        recall (float): Recall/Sensitivity (TP / (TP + FN))
        precision (float): Precision (TP / (TP + FP))
        f1_score (float): F1-score
    """
    probs = torch.sigmoid(preds).view(-1)
    labels = labels.view(-1)
    preds_bin = (probs >= threshold).float()
    
    tp = ((preds_bin == 1) & (labels == 1)).float().sum()
    fn = ((preds_bin == 0) & (labels == 1)).float().sum()
    fp = ((preds_bin == 1) & (labels == 0)).float().sum()
    
    # accuracy
    accuracy = (preds_bin == labels).float().mean()
    # recall
    recall = tp / (tp + fn + 1e-8)
    # precision
    precision = tp / (tp + fp + 1e-8)
    # f1-score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8) 
    
    return accuracy.item(), recall.item(), precision.item(), f1_score.item()

