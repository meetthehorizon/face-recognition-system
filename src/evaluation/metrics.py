import torch 
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix   
from sklearn.preprocessing import label_binarize

def compute_metrics(y_true, y_score, metrics=['accuracy','f1','precision','recall', 'confusion_matrix']):
    y_true = y_true.cpu().detach().numpy()
    y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
    y_score_detached = y_score.cpu().detach().numpy()
    softmax = nn.Softmax(dim=1)
    y_prob = softmax(torch.from_numpy(y_score_detached))
    y_pred = np.argmax(y_prob, axis=1)
    # y_true = y_true.reshape(1, -1)
    # print(f"Score: {y_score_detached.shape}")
    # print(f"y_pred: {y_pred.shape}")
    # print(f"y_true: {y_true.shape}")
    y_pred = y_pred.numpy()
    # print(f"Type :{type(y_true)}")
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
        if metric == 'accuracy':
            results[metric] = accuracy_score(y_true, y_pred)
        elif metric == 'confusion_matrix':
            results[metric] = confusion_matrix(y_true, y_pred)
            sns.heatmap(results[metric], annot=True, fmt='g')
            plt.show()
        elif metric == 'f1':
            results[metric] = f1_score(y_true, y_pred, average='micro')
        elif metric == 'precision':
            results[metric] = precision_score(y_true, y_pred, average='micro')
        elif metric == 'recall':
            results[metric] = recall_score(y_true, y_pred, average='micro')
        elif metric == 'roc_auc_score':
            results[metric] = roc_auc_score(y_true_binarized, y_prob, multi_class='ovr', average='macro')
        else:
            raise ValueError(f'Unsupported metric {metric}')
    return results
    