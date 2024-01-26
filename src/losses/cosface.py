import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import vector_norm as norm


class CosFaceLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        feat_dim,
        margin=0.35,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.feat_dim = feat_dim

        self.weight = nn.Parameter(torch.FloatTensor(feat_dim, num_classes))
        nn.init.xavier_uniform_(self.weight)

        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        # x-shape is (batch_num, feat_dim)
        # w-shape is (num_classes, feat_dim)
        x_norm = norm(x, ord=2, dim=1, keepdim=True)
        x_norm = x_norm.repeat(1, self.num_classes)
        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=0)

        cosine = F.linear(x, torch.t(W), bias=None)
        phi = cosine - self.margin
        one_hot = F.one_hot(y, num_classes=self.num_classes)
        one_hot = torch.squeeze(one_hot, dim=1).float()
        y_hat = one_hot * phi + (1.0 - one_hot) * cosine
        y_hat *= x_norm

        loss = self.cross_loss(y_hat, one_hot)

        return loss


def test():
    loss = CosFaceLoss(num_classes=10, feat_dim=2)
    x = torch.randn(2, 2)
    y = torch.tensor([0, 1])
    out = loss(x, y)


if __name__ == "__main__":
    print("passed")
    test()
