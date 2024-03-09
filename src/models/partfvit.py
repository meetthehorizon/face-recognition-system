import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cosineprod import MarginCosineProduct
from src.models.mobilenetv3 import CustomMobileNetV3


class PartfVit(nn.Module):
    def __init__(
        self,
        num_ids: int,
        image_size: int,
        mobilenet_size: str,
        num_landmarks: int,
        patch_size: int,
        mlp_dim: int,
        feat_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        eps: float,
    ):
        super().__init__()

        self.eps = eps
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_landmarks = num_landmarks

        self.lin_embedding = nn.Linear(3 * patch_size**2, feat_dim, bias=False)
        self.pos_embedding = nn.Parameter(torch.randn(num_landmarks, feat_dim))

        self.cls_token = nn.Parameter(torch.randn(feat_dim))

        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feat_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.cos_prod = MarginCosineProduct(feat_dim, num_ids)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.5)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.5)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, batch, pred):
        batch_size = batch.shape[0]
        landmarks = self._get_patches(self.image_size, self.num_landmarks)
        landmarks = landmarks.to(batch.device).repeat([batch_size, 1])

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
        batch = self._extract_landmarks(
            batch=batch,
            landmarks=landmarks,
            patch_size=[self.patch_size, self.patch_size],
        )  # (batch_size, num_landmarks, in_channels, patch_size, patch_size)

        batch = batch = batch.view(-1, self.num_landmarks, 3 * self.patch_size**2)
        batch = self.lin_embedding(batch) + self.pos_embedding

        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)

        z0 = torch.cat((cls_tokens, batch), dim=1)
        zL = self.layers(z0)

        y_score = zL[:, 0, :]

        logits = self.cos_prod(y_score, pred)
        return logits

    @staticmethod
    def _extract_landmarks(batch, landmarks, patch_size):

        num_landmarks = landmarks.shape[1] // 2
        batch_size, channels, image_height, image_width = batch.shape
        patch_height, patch_width = patch_size

        landmarks = landmarks.view(
            batch_size, num_landmarks, 2
        )  # Reshaping landmarks to (batch_size, num_landmarks, 2)

        device = landmarks.device

        patch_range = [
            [-patch_size[0] / 2, patch_size[0] / 2],
            [-patch_size[1] / 2, patch_size[1] / 2],
        ]  # the x and y range of the patches

        grid_x, grid_y = torch.meshgrid(
            torch.arange(patch_range[0][0], patch_range[0][1]),
            torch.arange(patch_range[1][0], patch_range[1][1]),
            indexing="ij",
        )  # generating the grid of x and y values shape: (patch_height, patch_width)

        sampling_grid = torch.stack(
            (-grid_y, grid_x), dim=-1
        )  # shape: (patch_height, patch_width, 2)

        sampling_grid = sampling_grid.to(device)

        list_patches = []

        for i in range(num_landmarks):
            land = landmarks[:, i, :]
            patch_grid = (land[:, None, None, :] + sampling_grid[None, :, :, :]) / (
                0.5
                * torch.tensor(
                    [image_height, image_width], dtype=torch.float32, device=device
                )[None, None, None, :]
            ) - 1
            single_landmark_patch = F.grid_sample(
                batch, patch_grid, align_corners=False
            )
            list_patches.append(single_landmark_patch)

        patches = torch.stack(list_patches, dim=1)
        return patches

    @staticmethod
    def _get_patches(image_size, num_landmarks):
        num_per_side = int(math.sqrt(num_landmarks))
        temp = (
            torch.linspace(0.0, float(image_size), num_per_side + 2)[1:-1]
            .unsqueeze(dim=0)
            .repeat((num_per_side, 1))
        )

        y = temp.flatten()
        x = temp.T.flatten()

        bias = torch.empty(2 * num_landmarks)
        bias[0::2], bias[1::2] = x, y

        return bias


if __name__ == "__main__":
    model = vit_b_32()

    x = torch.rand(2, 3, 224, 224)
    y = torch.rand(2).long()

    out = model(x, y)
    criterion = torch.nn.CrossEntropyLoss()
