import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader

from src.data.data_loader import DigiFace
from src.losses.cosface import CosFaceLoss
from src.models.resnet50 import ResNet50
from src.models.transformer import Transformer
from src.utils.pytorch_utils import extract_landmarks_from_image


class PartFVitWithLandmark(nn.Module):
    """Part-fVit with landmarks

    reference: https://arxiv.org/pdf/2212.00057v1.pdf
    """

    def __init__(
        self,
        num_identites,
        num_landmarks=49,
        patch_size=28,
        in_channels=3,
        image_size=112,
        feat_dim=768,
        mlp_dim=2048,
        num_heads=12,
        num_layers=12,
        dropout=0.0,
    ):
        """
        Parameters
        ----------
        num_identites : int
                Number of classes in dataset
        num_landmarks : int
                Number of landmarks to learn
        patch_size : int
                Size of patch extracted from image
        in_channels : int
                Number of input channels
        image_size : int
                Size of input image
        feat_dim : int
                Dimension of hidden feature vector
        mlp_dim : int
                Dimension of MLP in transformer
        num_heads : int
                Number of heads in transformer
        num_layers : int
                Number of layers of MSA and MLP used
        dropout : float
                Dropout rate
        """
        super().__init__()
        self.num_landmarks = num_landmarks
        self.image_size = image_size
        self.eps = 1e-05
        self.patch_dim = in_channels * patch_size * patch_size
        self.feat_dim = feat_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_identites = num_identites
        self.patch_size = patch_size

        self.cls_token = torch.nn.Parameter(
            torch.randn(self.feat_dim)
        )  # classificaiton token
        self.pos_embedding = torch.nn.Parameter(
            torch.randn(num_landmarks + 1, self.feat_dim)
        )  # positional embedding

        self.landmark_CNN = ResNet50(
            img_channels=in_channels, num_classes=2 * num_landmarks
        )  # landmark extractor

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
        )

        self.layers = Transformer(
            feat_dim=self.feat_dim,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

    def forward(self, batch):
        """Forward pass for part-fVit with landmarks

        Parameters
        ----------
        batch : torch.tensor
                Batch of images of shape (batch_size, in_channels, image_height, image_width)

        Returns
        -------
        torch.tensor
                Feature vector of shape (batch_size, feat_dim) the first token of which is the classification token and can be used for classification
        """
        batch_size = batch.shape[0]
        landmarks = self.landmark_CNN(batch)  # (batch_size, 2 * num_landmarks)

        # uniform scaling of landmarks to [0, image_size)
        temp_max = torch.max(landmarks, dim=1)[0]
        temp_max = torch.unsqueeze(temp_max, dim=1).repeat([1, 2 * self.num_landmarks])

        temp_min = torch.min(landmarks, dim=1)[0]
        temp_min = torch.unsqueeze(temp_min, dim=1).repeat([1, 2 * self.num_landmarks])

        landmarks = (
            (landmarks - temp_min)
            / (temp_max - temp_min + self.eps)
            * (self.image_size - 1)
        )  # (batch_size, 2 * num_landmarks)

        # extract patches from image using bilinear interpolation
        batch = extract_landmarks_from_image(
            batch=batch,
            landmarks=landmarks,
            patch_size=[self.patch_size, self.patch_size],
        )  # (batch_size, num_landmarks, in_channels, patch_size, patch_size)

        # flattening and converting dimesion to feat_dim
        batch = batch.view(batch_size, self.num_landmarks, self.patch_dim)
        batch = self.to_patch_embedding(batch)

        # adding cls token and positional embedding
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        z0 = torch.cat((cls_tokens, batch), dim=1)
        z0 += self.pos_embedding

        # returning final layer after transformer
        output = self.layers(z0)[:, 0, :]

        return output


if __name__ == "__main__":
    print("passed")
