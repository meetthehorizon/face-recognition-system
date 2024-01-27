import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


def plot_confusion_matrix(y_actual, y_hat):
    """Plots the confusion matrix for given actual and predicted values
    Parameters
    ----------
    y_actual : array-like
        The actual values
    y_hat : array-like
    """

    sns.heatmap(confusion_matrix(y_actual, y_hat), annot=True, cmap="prism", fmt="g")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title("Confusion Matrix")
    plt.show()


def plot_roc_curve(y_actual, y_hat):
    """Plots the ROC curve for given actual and predicted values
    Parameters
    ----------
    y_actual : array-like
        The actual values
    y_hat : array-like
    """

    fpr, tpr, _ = roc_curve(y_actual, y_hat)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


def plot_loss(history):
    """Plots the loss for given history
    Parameters
    ----------
    history : History
        The history object
    """

    plt.plot(history.history["loss"], color="blue")
    plt.plot(history.history["val_loss"], color="red")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper right")
    plt.show()


def plot_accuracy(history):
    """Plots the accuracy for given history
    Parameters
    ----------
    history : History
        The history object
    """
    plt.plot(history.history["accuracy"], color="purple")
    plt.plot(history.history["val_accuracy"], color="green")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="lower right")
    plt.show()


def plot_network(model):
    """Plots the network for given model
    Parameters
    ----------
    model : Model
        The model object
    """

    plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)


if __name__ == "__main__":
    plot_network(DigiFace)
