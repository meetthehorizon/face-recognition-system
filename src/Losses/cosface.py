import torch
import torch.nn as nn
import torch.nn.functional as F


# class CosFaceLoss(nn.Module):
#     def __init__(
#         self,
#         num_classes,
#         feat_dim,
#         margin=0.35,
#     ):
#         super().__init__()
#         self.num_classes = num_classes
#         self.margin = margin
#         self.feat_dim = feat_dim

#         self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))

#     def forward(self, x, y):
#         # x-shape is (batch_num, feat_di
        # w-shape is (num_classes, feat_dim)m)
#         x_norm = vector_norm(x, ord=2, dim=1, keepdim=True)
#         x = F.normalize(x, p=2, dim=
        w_norm = norm(self.weight, ord=2, dim=1, keepdim=True)
sdklfnk
#      dgnksiolaseodgjlsadkfeoilkm m)

#         pass

