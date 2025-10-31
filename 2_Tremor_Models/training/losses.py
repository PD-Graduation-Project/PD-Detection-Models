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
                focal_weight=0.8,        
                tversky_weight=1.0,      
                
                # classes params
                class_weights=torch.tensor([1.85, 0.53, 1.28]),
                num_classes=3, 
                
                # focal params
                focal_alpha=0.8,
                focal_gamma=2.0,
                
                # tversky params
                tversky_alpha=0.6,
                tversky_beta=0.4,
                
                # NEW: Adaptive parameters
                adaptive_weighting=True,   # Enable dynamic loss weighting
                label_smoothing=0.1,      # ENHANCED: increased smoothing
                ):
        super().__init__()
        
        # 1. init all params
        # -------------------
        
        # 1.1. Final combined loss weights
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        
        # 1.2. Optional per-class weights (for imbalance)
        self.register_buffer('class_weights', class_weights)
        self.num_classes = num_classes
        
        # 1.3. Focal loss params
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # 1.4. Tversky loss params
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        
        # 1.5. NEW: Adaptive parameters
        self.adaptive_weighting = adaptive_weighting
        self.label_smoothing = label_smoothing
        
        # 1.6. NEW: Loss tracking for adaptive weighting (will be moved to device automatically)
        self.ce_running = None
        self.focal_running = None
        self.tversky_running = None
        self.update_count = 0
        
    # 2. Forward function
    # -----------------------
    def forward(self, pred, label):
        """
        pred: [B, num_classes] (logits)
        label: [B] (class indices)
        """
        # NEW: Initialize running averages on correct device if not done
        if self.ce_running is None:
            device = pred.device
            self.ce_running = torch.zeros(1, device=device)
            self.focal_running = torch.zeros(1, device=device)
            self.tversky_running = torch.zeros(1, device=device)
            
        # NEW: Apply label smoothing to all components
        if self.label_smoothing > 0:
            label_one_hot = F.one_hot(label, self.num_classes).float()
            label_one_hot = label_one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
        else:
            label_one_hot = F.one_hot(label, self.num_classes).float()
            
        # 1. CrossEntropy Loss (multi-class)
        # ------------------------------------
        ce = F.cross_entropy(pred, label, weight=self.class_weights,
                            label_smoothing= self.label_smoothing)

        # 2. Focal Loss (multi-class)
        # -------------------------------
        probs = F.softmax(pred, dim=1) # [B, C]
        # NEW: Class-balanced focal alpha
        if isinstance(self.focal_alpha, (list, torch.Tensor)):
            alpha_t = self.focal_alpha[label]  # Per-sample alpha
        else:
            alpha_t = self.focal_alpha
        pt = torch.sum(label_one_hot * probs, dim=1) + 1e-8 # prob of true class
        focal = -alpha_t * ((1 - pt) ** self.focal_gamma) * torch.log(pt)
        focal = focal.mean()

        # 3. Tversky Loss (multi-class)
        # -------------------------------
        probs_flat = probs.view(-1, self.num_classes)
        labels_flat = label_one_hot.view(-1, self.num_classes)
        
        TP = (probs_flat * labels_flat).sum(dim=0)
        FP = ((1 - labels_flat) * probs_flat).sum(dim=0)
        FN = (labels_flat * (1 - probs_flat)).sum(dim=0)
        
        # NEW: Add smoothness to prevent division by zero and improve gradient flow
        tversky_index = (TP + 1.0) / (TP + self.tversky_alpha * FN + self.tversky_beta * FP + 1.0)
        tversky = 1 - tversky_index.mean()
        
        
        # NEW: 4. Adaptive loss weighting based on current magnitudes
        # -------------------------------------------------------------
        if self.adaptive_weighting and self.training:
            # Update running averages
            self.ce_running = 0.9 * self.ce_running + 0.1 * ce.detach()
            self.focal_running = 0.9 * self.focal_running + 0.1 * focal.detach()
            self.tversky_running = 0.9 * self.tversky_running + 0.1 * tversky.detach()
            self.update_count += 1
            
            # Avoid division by zero in early steps
            if self.update_count > 10:
                ce_scale = self.ce_running.mean()
                focal_scale = self.focal_running.mean()
                tversky_scale = self.tversky_running.mean()
                
                # Normalize weights by their relative scales
                total_scale = ce_scale + focal_scale + tversky_scale + 1e-8
                ce_weight = self.ce_weight * (total_scale / (ce_scale + 1e-8))
                focal_weight = self.focal_weight * (total_scale / (focal_scale + 1e-8))
                tversky_weight = self.tversky_weight * (total_scale / (tversky_scale + 1e-8))
                
                # Normalize to maintain overall scale
                weight_sum = ce_weight + focal_weight + tversky_weight + 1e-8
                ce_weight = ce_weight / weight_sum * 3.0  # Maintain total weight ~3.0
                focal_weight = focal_weight / weight_sum * 3.0
                tversky_weight = tversky_weight / weight_sum * 3.0
            else:
                ce_weight, focal_weight, tversky_weight = self.ce_weight, self.focal_weight, self.tversky_weight
        else:
            ce_weight, focal_weight, tversky_weight = self.ce_weight, self.focal_weight, self.tversky_weight
        
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