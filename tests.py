import torch 
import torch.nn as nn
import torch.nn.functional as F
from src.models import predict_model, transformer, train_model
from src.data import data_loader
from src.losses import loss

# PredictModel will return a tensor of size batch_size x 1
# This tensor will contain the predicted class for each image in the batch

def test_predict_model(batch_size = 1):
    assert(predict_model.PredictModel(torch.randn(1, 3, 112, 112)).shape == torch.Size([1, 1]))

def test_preprocess(batch_size = 1):
    assert(len(data_loader.DigiFace) == batch_size * 2)

def test_transformer(batch_size = 1):
    assert(transformer.Transformer)

