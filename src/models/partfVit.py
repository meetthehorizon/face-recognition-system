import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from random import randint

from src.data.data_loader import DigiFace

class part_fVit_with_landmark(nn.Module):
    def __init__(self):
        pass

    @staticmethod
    def extract_landmarks_from_image(batch, landmarks, patch_size):
        """Extract landmarks from image.

        Parameters
        ----------
        batch : torch.Tensor
                Batch of images with shape (batch_size, channels, height, width)
        landmarks : torch.Tensor
                Batch of landmarks with flattened co-ordinates of shape (batch_size, 2 * n_landmarks)
        patch_size : list
                Size of patches to extract of form [height, width]

        Returns
        -------
        patches : torch.Tensor
                Batch of patchs with shaepe (batch_size, n_landmarks, channels, height, width))
        """

        n_landmarks = landmarks.shape[1] // 2
        batch_size, channels, image_height, image_width = batch.shape
        patch_height, patch_width = patch_size

        landmarks = landmarks.view(
            batch_size, n_landmarks, 2
        )  # Reshaping landmarks to (batch_size, n_landmarks, 2)

        patch_range = [
            [-patch_size[0] / 2, patch_size[0] / 2],
            [-patch_size[1] / 2, patch_size[1] / 2],
        ]  # the x and y range of the patches

        grid_x, grid_y = torch.meshgrid(
            torch.arange(patch_range[0][0], patch_range[0][1]),
            torch.arange(patch_range[1][0], patch_range[1][1]),
            indexing="ij",
        )  # generating the grid of x and y values shape: (patch_height, patch_width)

        sampling_grid = torch.stack((-grid_y, grid_x), dim=-1)  # shape: (patch_height, patch_width, 2)
        list_patches = []

        for i in range(n_landmarks):
            land = landmarks[:, i, :]
            patch_grid = (land[:, None, None, :] + sampling_grid[None, :, :, :]) / (
                0.5
                * torch.tensor([image_height, image_width], dtype=torch.float32)[
                    None, None, None, :
                ]
            ) - 1
            single_landmark_patch = F.grid_sample(
                batch, patch_grid, align_corners=False
            )
            list_patches.append(single_landmark_patch)

        patches = torch.stack(list_patches, dim=1)
        return patches

def display_images(tensor):
    # Assuming tensor has dimensions [num, channels, height, width]
    num_images, num_channels, height, width = tensor.size()

    # Reshape the tensor to [num, height, width, channels]
    tensor = tensor.permute(0, 2, 3, 1)

    # Display each image in a subplot
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(tensor[i].cpu().numpy())  # Assuming the tensor is on the CPU
        plt.axis('off')

    plt.show()

if __name__ == "__main__":
    print("testing script")
    dataset = DigiFace(transform=transforms.ToTensor())

    batch = dataset[randint(0, len(dataset))][0]
    batch = batch.view(1, 3, 112, 112)
    landmarks = torch.tensor([[55, 55, 84, 84]])
    patch_size = [28, 28]

    model = part_fVit_with_landmark()

    patch = model.extract_landmarks_from_image(
        batch=batch, landmarks=landmarks, patch_size=patch_size
    )
    display_images(patch[0])
