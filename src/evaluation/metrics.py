import torch 
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, roc_curve, f1_score, precision_score, recall_score

def compute_metrics(y_true, y_pred, y_score, metrics):
    """
    Compute metrics for a given set of predictions.
    Args:
        y_true: true labels
        y_pred: predicted labels
        y_score: predicted scores
        metrics: list of metrics to compute
    Returns:
        Dictionary of metric names and values
    """
    results = {}
    for metric in metrics:
        if metric == 'roc_auc':
            results[metric] = roc_auc_score(y_true, y_score)
        elif metric == 'pr_auc':
            results[metric] = average_precision_score(y_true, y_score)
        elif metric == 'pr_curve':
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            results[metric] = auc(recall, precision)
        elif metric == 'roc_curve':
            fpr, tpr, _ = roc_curve(y_true, y_score)
            results[metric] = auc(fpr, tpr)
        elif metric == 'accuracy':
            results[metric] = (y_true == y_pred).mean()
        elif metric == 'f1':
            results[metric] = f1_score(y_true, y_pred)
        elif metric == 'precision':
            results[metric] = precision_score(y_true, y_pred)
        elif metric == 'recall':
            results[metric] = recall_score(y_true, y_pred)
        else:
            raise ValueError(f'Unsupported metric {metric}')
    return results