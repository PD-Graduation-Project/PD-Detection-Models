import torch
from torch import nn
import torch.nn.functional as F


# Weighted Combined Losses class
# -------------------------------
class CombinedLoss(nn.Module):
    """
    Combined CrossEntropy + Focal + Tversky Loss for multi-class classification (3 classes).
    
    Final Loss = w1*CE + w2*Focal + w3*Tversky
    
    This loss blends three components:
      - **CrossEntropy:** Standard classification loss (supports class weighting).
      - **Focal:** Emphasizes hard-to-classify samples using `focal_alpha` and `focal_gamma`.
      - **Tversky:** Handles class imbalance and recall-precision tradeoff with `tversky_alpha`, `tversky_beta`.

    Designed to emphasize recall for under-represented or hard classes.
    
    Args:
        class_weights (torch.Tensor, optional): Weights for each class (default: [1.85, 0.53, 1.28]).
        focal_alpha (float): Balance factor for Focal loss (default: 0.25).
        focal_gamma (float): Focusing parameter for Focal loss (default: 2.0).
        tversky_alpha (float): Weight for false negatives in Tversky (default: 0.5).
        tversky_beta (float): Weight for false positives in Tversky (default: 0.5).
        weights (tuple[float]): Relative weights of the three losses (default: (1.0, 1.0, 1.0)).

    Returns:
        torch.Tensor: Combined scalar loss.
        
    > For more details on hyperparameters effects see the docstring in the 
    loss function in the spiral drawings model.
    """
    def __init__(self,
                # losses weights (final equation)
                ce_weight=1.0,
                focal_weight=0.5,
                tversky_weight=0.5,
                
                # classes params
                class_weights=torch.tensor([1.85, 0.53, 1.28]),
                num_classes=3, 
                
                # focal params
                focal_alpha=0.6,
                focal_gamma=2.0,
                
                # tversky params
                tversky_alpha=0.7,
                tversky_beta=0.3):
        super().__init__()
        
        # 1. init all params
        # -------------------
        
        # 1.1. Final combined loss weights
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        
        # 1.2. Optional per-class weights (for imbalance)
        self.class_weights = class_weights
        self.num_classes = num_classes
        
        # 1.3. Focal loss params
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # 1.4. Tversky loss params
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        
    # 2. Forward function
    # -----------------------
    def forward(self, pred, label):
        """
        pred: [B, num_classes] (logits)
        label: [B] (class indices)
        """
        # 1. CrossEntropy Loss (multi-class)
        ce = F.cross_entropy(pred, label, weight=self.class_weights)

        # 2. Focal Loss (multi-class)
        probs = F.softmax(pred, dim=1)                     # [B, C]
        label_one_hot = F.one_hot(label, self.num_classes).float()  # [B, C]
        pt = torch.sum(label_one_hot * probs, dim=1) + 1e-8 # prob of true class
        focal = -self.focal_alpha * ((1 - pt) ** self.focal_gamma) * torch.log(pt)
        focal = focal.mean()

        # 3. Tversky Loss (multi-class)
        probs_flat = probs.view(-1, self.num_classes)
        labels_flat = label_one_hot.view(-1, self.num_classes)
        
        TP = (probs_flat * labels_flat).sum(dim=0)
        FP = ((1 - labels_flat) * probs_flat).sum(dim=0)
        FN = (labels_flat * (1 - probs_flat)).sum(dim=0)
        
        tversky_index = (TP + 1.0) / (TP + self.tversky_alpha * FN + self.tversky_beta * FP + 1.0)
        tversky = 1 - tversky_index.mean()
        
        # Combine all
        total_loss = (self.ce_weight * ce +
                      self.focal_weight * focal +
                      self.tversky_weight * tversky)
        
        return total_loss


# Upgraded metrics function
# ---------------------------
def multiclass_metrics(preds, labels, num_classes=3):
    """
    Compute multi-class accuracy, recall, precision, and F1-score.
    
    Uses **Macro averages**, where the metric is computed per class and then averaged equally
    across all classes, giving each class the same weight regardless of how many
    samples it has (equal importance to each class). This helps evaluate performance on imbalanced datasets.

    Args:
        preds (Tensor): [B, num_classes] logits or probabilities.
        labels (Tensor): [B] ground-truth class indices.
        num_classes (int): total number of classes.

    Returns:
        dict: {
            'accuracy': float,
            'recall_macro': float,
            'precision_macro': float,
            'f1_macro': float,
            'recall_per_class': list[float],
            'precision_per_class': list[float],
            'f1_per_class': list[float]
        }
    """
    # Convert logits -> predicted class
    preds_cls = preds.argmax(dim=1)
    labels = labels.view(-1)

    # Overall accuracy
    accuracy = (preds_cls == labels).float().mean().item()

    # Initialize containers
    recall_list, precision_list, f1_list = [], [], []

    # Per-class metrics
    for c in range(num_classes):
        tp = ((preds_cls == c) & (labels == c)).sum().float()
        fn = ((preds_cls != c) & (labels == c)).sum().float()
        fp = ((preds_cls == c) & (labels != c)).sum().float()

        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        recall_list.append(recall.item())
        precision_list.append(precision.item())
        f1_list.append(f1.item())

    # Macro averages
    recall_macro = sum(recall_list) / num_classes
    precision_macro = sum(precision_list) / num_classes
    f1_macro = sum(f1_list) / num_classes

    return {
        'accuracy': accuracy,
        'recall_macro': recall_macro,
        'precision_macro': precision_macro,
        'f1_macro': f1_macro,
        'recall_per_class': recall_list,
        'precision_per_class': precision_list,
        'f1_per_class': f1_list
    }