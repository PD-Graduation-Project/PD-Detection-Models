import torch
from torch import nn
import torch.nn.functional as F

# Weighted Combined Losses class
# -------------------------------
# Weighted Combined Losses class
# -------------------------------
class CombinedLoss(nn.Module):
    """
    Combined BinaryCrossEntropy + Focal + Tversky Loss for imbalanced binary classification.
    
    Designed for: 77% Parkinson (majority) vs 23% Healthy (minority)
    
    Final Loss = w1*BCE + w2*Focal + w3*Tversky

    PARAMETER GUIDE - WHAT EACH DOES:
    ==================================
    
    LOSS COMPONENT WEIGHTS (balance between loss types):
    ----------------------------------------------------
    - bce_weight: Weight for standard BCE loss (stable baseline)
    - focal_weight: Weight for Focal loss (focuses on hard examples)
    - tversky_weight: Weight for Tversky loss (controls FN vs FP trade-off)
    
    CLASS IMBALANCE CORRECTION:
    ---------------------------
    - healthy_weight: Penalty multiplier for missing Healthy samples (minority class)
                    ↑ higher = model pays more attention to Healthy class
                    For 77/23 split: use ~3.3 (ratio: 0.77/0.23)
                    
    - parkinson_weight: Penalty multiplier for missing Parkinson samples (majority class)
                        Usually keep at 1.0 since PD is already 77% of data
                        Only increase if PD recall is too low
    
    FOCAL LOSS PARAMS (handles hard-to-classify examples):
    -------------------------------------------------------
    - focal_alpha: Which class to focus on
                < 0.5 = focus on MINORITY (Healthy) - RECOMMENDED for imbalanced data
                > 0.5 = focus on MAJORITY (Parkinson)
                = 0.5 = balanced
                For 77/23 split: use 0.23 (match minority proportion)
                
    - focal_gamma: How much to focus on hard examples
                = 0: regular BCE (no focusing)
                = 2: standard (moderate focusing) - RECOMMENDED
                > 2: aggressive focusing on hardest examples
                Higher = model focuses more on mistakes
    
    TVERSKY LOSS PARAMS (precision vs recall trade-off):
    -----------------------------------------------------
    - tversky_alpha: False Negative penalty (controls RECALL)
                    ↑ higher = punish missing positives more = HIGHER RECALL
                    Lower = allow more FN = lower recall, higher precision
                    
    - tversky_beta: False Positive penalty (controls PRECISION)
                    ↑ higher = punish false alarms more = HIGHER PRECISION
                    Lower = allow more FP = higher recall, lower precision
                    
    Recommended for balanced F1: alpha + beta = 1.0
    - For recall focus: alpha=0.7, beta=0.3
    - For balanced: alpha=0.5, beta=0.5
    - For precision focus: alpha=0.3, beta=0.7
    
    COMMON SCENARIOS:
    -----------------
    1. Model predicts everything as Parkinson (high recall, low precision):
        - Increase healthy_weight (e.g., 3.5 → 4.5)
        - Decrease focal_alpha (e.g., 0.25 → 0.20)
        - Increase tversky_beta for precision (e.g., 0.3 → 0.4)

    2. Model misses too many Parkinson cases (low recall):
        - Increase tversky_alpha (e.g., 0.7 → 0.8)
        - Increase focal_gamma (e.g., 2 → 3)
        
    3. Model is unstable/overfitting:
        - Decrease focal_weight (e.g., 1.0 → 0.5)
        - Increase bce_weight (e.g., 0.5 → 0.8)
    """
    
    def __init__(self,
                # Loss component weights
                bce_weight=0.5,
                focal_weight=1.0,
                tversky_weight=0.5,
                
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
        
        # 1. Binary Cross-Entropy with class weighting
        # ----------------------------------------------
        # Apply higher weight to minority class (Healthy=0)
        weights = torch.where(
            label == 1,  # Parkinson (majority)
            self.parkinson_weight.to(device),
            self.healthy_weight.to(device)  # Healthy (minority) - higher weight
        )
        
        # FIXED: Use logits directly (no double sigmoid)
        bce = F.binary_cross_entropy_with_logits(pred, label, weight=weights)
        
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
