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

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, y):
        # x-shape is (batch_num, feat_dim)
        # w-shape is (num_classes, feat_dim)
        x_norm = norm(x, ord=2, dim=1, keepdim=True)
        x = F.normalize(x, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(x, W, bias=None)
        phi = cosine - self.margin
        one_hot = F.one_hot(y, num_classes=self.num_classes)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= x_norm
        output = F.cross_entropy

        return output


def main():
    loss = CosFaceLoss(num_classes=10, feat_dim=2)
    x = torch.randn(2, 2)
    y = torch.tensor([0, 1])
    out = loss(x, y)
    print(out)


if __name__ == "__main__":
    print("testing script")
    main()
