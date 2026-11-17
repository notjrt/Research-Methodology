import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

def compute_classification_metrics(y_true, y_pred, y_prob=None, average="macro"):
    """
    Compute comprehensive scientific classification metrics.
    
    Returns:
        dict: Dictionary containing accuracy, macro sensitivity/recall,
              macro precision, macro specificity, and macro F1 score.
    """
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    
    # Calculate per-class specificity and sensitivity
    specificities = []
    sensitivities = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        specificities.append(spec)
        sensitivities.append(sens)
        
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_sensitivity": float(np.mean(sensitivities)),
        "macro_specificity": float(np.mean(specificities)),
        "macro_precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "confusion_matrix": cm.tolist()
    }
    
    if y_prob is not None:
        try:
            metrics["macro_roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            metrics["macro_roc_auc"] = None
            
    return metrics
