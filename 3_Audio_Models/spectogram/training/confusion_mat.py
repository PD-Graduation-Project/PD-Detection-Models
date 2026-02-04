import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(model, dataloader, device, class_names=None, threshold=0.5):
    """
    Computes and plots the confusion matrix for a binary/multi-class classification model.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    dataloader : torch.utils.data.DataLoader
        DataLoader for validation/test data.
    device : torch.device
        Device to run inference on.
    class_names : list[str], optional
        List of class names for labels.
    threshold : float, optional
        Threshold for converting probabilities to binary predictions (default=0.5)
    """
    model.eval().to(device)
    all_preds = []
    all_labels = []
    
    with torch.inference_mode():
        for batch in dataloader:
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # forward pass
            logits = model(imgs)
            if isinstance(logits, tuple):
                logits = logits[0]
            
            # convert to probabilities if binary classification
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).long()
            else:
                preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    # concatenate all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # plot
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
