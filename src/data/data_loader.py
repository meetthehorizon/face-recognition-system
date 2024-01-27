import os
import torch

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image


class DigiFace(Dataset):
    """DigiFace dataset class

    Attributes
    ----------
    path : str
       Path to images directory
    transform : callable
       Transformation to apply to images
    """

    def __init__(
        self, path="data/raw", transform=transforms.ToTensor(), num_identities=None
    ):
        """
        Parameters
        ----------
        path : str
                Path to images directory
        transform : callable
                Transformation to apply to image and mask
        num_identities : int
                Number of identities to use from dataset
        """
        self.path = path
        self.transform = transform
        self.identities = os.listdir(path)
        self.num_identities = num_identities
        self.samples = self._generate_samples(num_identities=num_identities)

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : int
                Index of sample to load

        Returns
        -------
        (image, label) : tuple
                image : torch.Tensor
                        Image at index
                label : int
                        Label of image at index
        """
        if index >= len(self) or index < 0:
            raise IndexError("Index out of range")

        identity, image_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, int(identity)

    def _generate_samples(self, num_identities):
        """Generate sample from dataset.

        Parameters
        ----------
        num_identities : int
                Number of identities to use from dataset

        Returns
        -------
        samples : list
                path of images
        """
        samples = []
        for identity in self.identities:
            identity_path = os.path.join(self.path, identity)
            if num_identities:
                identity_path = identity_path[:num_identities]
            images_path = os.listdir(identity_path)

            for image_path in images_path:
                image_path = os.path.join(identity_path, image_path)
                samples.append((identity, image_path))

        return samples

    def __len__(self):
        """Return length of datase"""
        return len(self.samples)


if __name__ == "__main__":
    print("passed")
