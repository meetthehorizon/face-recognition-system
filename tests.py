import torch 
import torch.nn as nn
import torch.nn.functional as F
from src.models import predict_model

# PredictModel will return a tensor of size batch_size x 1
# This tensor will contain the predicted class for each image in the batch

def test_predict_model(batch_size = 1):
    assert(predict_model.PredictModel(torch.randn(1, 3, 112, 112)).shape == torch.Size([1, 1]))
