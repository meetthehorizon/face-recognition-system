import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms


def test_data_loader(DigiFace):
    transform = transforms.PILToTensor()
    dataset = DigiFace(path="./data/raw/", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = dataloader.__iter__().__next__()

    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    axes = axes.flatten()
    for i, (image, identity) in enumerate(zip(*batch)):
        image = image.permute(1, 2, 0)
        axes[i].imshow(image)
        axes[i].set_title(identity.item())
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def test_resnet50(ResNet50):
    BATCH_SIZE = 32
    net = ResNet50(img_channels=3, num_classes=1000)
    x = torch.randn(BATCH_SIZE, 3, 112, 112)
    y = net(x)
    assert y.size() == torch.Size([BATCH_SIZE, 1000])


def test_transformer(Transformer):
    BATCH_SIZE = 32
    transformer_model = Transformer(img_channels=3, num_classes=1000)
    x = torch.randn(BATCH_SIZE, 3, 112, 112)
    y = transformer_model(x)
    assert y.size() == torch.Size([BATCH_SIZE, Transformer.seq_len, Transformer.feat_dim])


def test_partfVit():
    pass

def test_predict_model(batch_size=1):
    pass