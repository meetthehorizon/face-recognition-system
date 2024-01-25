import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from random import randint

from src.data.data_loader import DigiFace
from src.models.resnet50 import ResNet50
from src.models.transformer import Transformer
from src.utils.pytorch_utils import extract_landmarks_from_image


class part_fVit_with_landmark(nn.Module):
    def __init__(
        self,
        num_landmarks=49,
        patch_size=28,
        in_channels=3,
        image_size=112,
        feat_dim=768,
        mlp_dim=2048,
        num_heads=11,
        num_layers=12,
        dropout=0.0,
    ):
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

        self.backbone = ResNet50(
            img_channels=in_channels, num_classes=2 * num_landmarks
        )
        self.patch_size = patch_size
        self.cls_token = torch.nn.Parameter(torch.randn(self.feat_dim))
        self.pos_embedding = torch.nn.Parameter(
            torch.randn(num_landmarks + 1, self.feat_dim)
        )

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
        )

    def forward(self, batch):
        batch_size, in_channels, image_heght, image_width = batch.shape
        landmarks = self.backbone(batch)

        temp_max = torch.max(landmarks, dim=1)[0]
        temp_max = torch.unsqueeze(temp_max, dim=1).repeat([1, 2 * self.num_landmarks])

        temp_min = torch.min(landmarks, dim=1)[0]
        temp_min = torch.unsqueeze(temp_min, dim=1).repeat([1, 2 * self.num_landmarks])

        landmarks = (
            (landmarks - temp_min)
            / (temp_max - temp_min + self.eps)
            * (self.image_size - 1)
        )

        batch = extract_landmarks_from_image(
            batch=batch,
            landmarks=landmarks,
            patch_size=[self.patch_size, self.patch_size],
        )

        batch = batch.view(batch_size, self.num_landmarks, self.patch_dim)
        batch = self.to_patch_embedding(batch)

        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        z0 = torch.cat((cls_tokens, batch), dim=1)

        z0 += self.pos_embedding

        return z0


def main():
    print("testing script")
    dataset = DigiFace(path="data/raw", transform=transforms.ToTensor(), num_images=8)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = part_fVit_with_landmark(num_landmarks=5)

    for batch in dataloader:
        images, labels = batch
        output = model(images)
        # with torch.no_grad():
        #   display_images(output[0])
        print(output.shape)
        break


if __name__ == "__main__":
    main()
