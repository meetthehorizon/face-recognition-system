import torch
import torch.nn as nn


class ConcatModelWithLoss(nn.Module):
    def __init__(self, main_model, criterion):
        super().__init__()
        self.main_model = main_model
        self.criterion = criterion

    def forward(self, batch, labels):
        cls_token = self.main_model(batch)
        y_pred = self.criterion(cls_token, labels)
        return y_pred
