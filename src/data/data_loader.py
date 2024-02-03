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
        self, path="data/raw", transform=transforms.PILToTensor(), num_identities=None
    ):
        """
        Parameters
        ----------
        path : str
                Path to images directory
        transform : callables
                Transformation to apply to image and mask
        num_identities : int
                Number of identities to use from dataset
        """
        self.path = path
        self.transform = transform
        self.path = path
        self.num_identities = (
            num_identities if num_identities is not None else len(os.listdir(path))
        )
        self.identities = sorted(os.listdir(path))[:num_identities]
        self.samples = self._generate_samples()

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

        image_path, identity = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image.float(), torch.tensor(identity, dtype=torch.long)

    def _generate_samples(self):
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
            for img_idx in os.listdir(identity_path):
                img_path = os.path.join(identity_path, img_idx)
                samples.append((img_path, int(identity)))
        return samples

    def __len__(self):
        """Return length of datase"""
        return len(self.samples)


if __name__ == "__main__":
    print("passed")
