import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from random import randint

from src.data.data_loader import DigiFace
from src.models.resnet50 import ResNet50


def extract_landmarks_from_image(batch, landmarks, patch_size):
    """Extract landmarks from image.

    Parameters
    ----------
    batch : torch.Tensor
            Batch of images with shape (batch_size, channels, height, width)
    landmarks : torch.Tensor
            Batch of landmarks with flattened co-ordinates of shape (batch_size, 2 * num_landmarks)
    patch_size : list
            Size of patches to extract of form [height, width]

    Returns
    -------
    patches : torch.Tensor
            Batch of patchs with shaepe (batch_size, num_landmarks, channels, height, width))
    """

    num_landmarks = landmarks.shape[1] // 2
    batch_size, channels, image_height, image_width = batch.shape
    patch_height, patch_width = patch_size

    landmarks = landmarks.view(
        batch_size, num_landmarks, 2
    )  # Reshaping landmarks to (batch_size, num_landmarks, 2)

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
    list_patches = []

    for i in range(num_landmarks):
        land = landmarks[:, i, :]
        patch_grid = (land[:, None, None, :] + sampling_grid[None, :, :, :]) / (
            0.5
            * torch.tensor([image_height, image_width], dtype=torch.float32)[
                None, None, None, :
            ]
        ) - 1
        single_landmark_patch = F.grid_sample(batch, patch_grid, align_corners=False)
        list_patches.append(single_landmark_patch)

    patches = torch.stack(list_patches, dim=1)
    return patches


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


def display_images(tensor):
    # Assuming tensor has dimensions [num, channels, height, width]
    num_images, num_channels, height, width = tensor.size()

    # Reshape the tensor to [num, height, width, channels]
    tensor = tensor.permute(0, 2, 3, 1)

    # Display each image in a subplot
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(tensor[i].cpu().numpy())  # Assuming the tensor is on the CPU
        plt.axis("off")

    plt.show()


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
