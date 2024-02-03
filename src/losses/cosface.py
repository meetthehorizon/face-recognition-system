import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.linalg import vector_norm as norm


class CosFaceLoss(nn.Module):
    """CosFace loss.

    Cos Face Loss for Part-fVit Architecture.
    Reference:
        - https://arxiv.org/pdf/1801.09414.pdf"""

    def __init__(
        self,
        num_classes,
        feat_dim,
        margin=0.35,
    ):
        """
        Parameters
        ----------
        num_classes : int
                Number of classes in dataset
        feat_dim : int
                Dimension of feature vector
        margin : float
                Margin for CosFace loss
        """

        super().__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.cross_loss = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(feat_dim, num_classes))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, y):
        """Forward pass for CosFace loss

        Parameters
        ----------
        x: torch.tensor
            The input feature vector of shape (batch_size, feat_dim)
        y: torch.tensor
            The ground truth class label of shape (batch_size, 1)

        """
        x_norm = norm(x, ord=2, dim=1, keepdim=True)
        x_norm = x_norm.repeat(1, self.num_classes)  # (batch_size, num_classes)

        # Normalisation of weights and features
        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=0)

        cosine = F.linear(x, torch.t(W), bias=None)  # (batch_size, num_classes)
        phi = cosine - self.margin
        one_hot = F.one_hot(
            y, num_classes=self.num_classes
        )  # One hot encoding of labels
        one_hot = torch.squeeze(one_hot, dim=1).float()

        # the following line creates margin for each correct class
        y_hat = one_hot * phi + (1.0 - one_hot) * cosine
        y_hat *= x_norm

        loss = self.cross_loss(y_hat, one_hot)

        return loss


if __name__ == "__main__":
    print("passed")
