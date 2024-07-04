import os
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(
    metrics_dict, y_true, y_score, metrics=["accuracy", "f1", "precision", "recall"]
):
    y_true = y_true.detach().cpu().numpy()
    y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
    y_score_detached = y_score.detach().cpu().numpy()
    softmax = nn.Softmax(dim=1)
    y_prob = softmax(torch.from_numpy(y_score_detached))
    y_pred = np.argmax(y_prob, axis=1)
    y_pred = y_pred.numpy()
    """
    Compute metrics for a given set of predictions.
    Args:
        metrics_dict: dictionary to store metric values
        y_true: true labels
        y_pred: predicted labels
        y_score: predicted scores
        metrics: list of metrics to compute
    Returns:
        Dictionary of metric names and values
    """
    for metric in metrics:
        if metric not in metrics_dict.keys():
            metrics_dict[metric] = list()
        if metric == "accuracy":
            metrics_dict[metric].append(accuracy_score(y_true, y_pred))
        elif metric == "f1":
            metrics_dict[metric].append(f1_score(y_true, y_pred, average="micro"))
        elif metric == "precision":
            metrics_dict[metric].append(
                precision_score(y_true, y_pred, average="micro")
            )
        elif metric == "recall":
            metrics_dict[metric].append(recall_score(y_true, y_pred, average="micro"))

    return metrics_dict


def save_confusion_matrix(cache, save_path, epoch):
    labels = []
    predictions = []

    for batch in cache:
        y_score, y_true = batch
        y_score = np.argmax(y_score, axis=1).astype(np.int64)
        labels.extend(y_true)
        predictions.extend(y_score)

    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")

    save_path = os.path.join(save_path, "cm")

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{epoch}.jpeg"))


def save_metrics(metrics, title, filename):
    epochs = range(1, len(metrics["losses"]) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, metrics["losses"], "b", label="Loss")
    plt.plot(epochs, metrics["accuracy"], "r", label="Accuracy")
    plt.plot(epochs, metrics["f1"], "g", label="F1 Score")
    plt.plot(epochs, metrics["precision"], "c", label="Precision")
    plt.plot(epochs, metrics["recall"], "m", label="Recall")

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()

    plt.savefig(filename)
