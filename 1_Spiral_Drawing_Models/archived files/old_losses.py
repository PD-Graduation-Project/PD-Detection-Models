from torch.nn import BCEWithLogitsLoss
from torch import nn, ones_like, sigmoid

# Weighted Binary Cross-Entropy Loss
# -------------------------------------
class WeightedBCE(nn.Module):
    """
    Weighted Binary Cross-Entropy loss for binary classification tasks.

    This loss extends BCEWithLogitsLoss by applying a higher penalty
    to false negatives (i.e., when the model predicts "Healthy" for PD cases).

    Args:
        FN_penalty (float): Multiplier for penalizing false negatives (label=1). 
                            Higher values make the model more sensitive to PD cases.

    Usage:
        loss_fn = WeightedBCE(FN_penalty=1.3)
        loss = loss_fn(pred, label)

    Expected input:
        pred:  Tensor of shape (N, 1): raw logits from the model.
        label: Tensor of shape (N, 1): binary ground-truth labels (0 or 1).
    """
    def __init__(self,
                FN_penalty:float = 1.3,):
        super().__init__()
        
        self.FN_penalty = FN_penalty
        self.bce  = BCEWithLogitsLoss(reduction='none')
        
        
    # Forward function
    # -----------------
    def forward(self, pred, label):
        # 1 calculate element-wise BCE loss
        bce_loss = self.bce (pred, label)
        
        # 2. create per-sample weights
        weights = ones_like(label)
        
        # 3. heavier penalty on PD (false negatives: label = 1)
        weights[label == 1] = self.FN_penalty 
        
        # 4. add those weights to the original loss and average
        weighted_loss  = (bce_loss*weights).mean()
        
        # 5. return loss
        return weighted_loss 
    
    
    
# Simple binary accuracy function
# --------------------------------
def binary_accuracy(preds, labels, threshold=0.5):
    """
    Computes binary accuracy for predictions vs. labels.
    Args:
        preds: raw logits from model
        labels: ground truth labels
        threshold: probability threshold for classification
    Returns:
        accuracy as a float tensor
    """
    probs = sigmoid(preds)
    preds_bin = (probs > threshold).float()
    return (preds_bin == labels).float().mean()