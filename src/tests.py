import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.data_loader import DigiFace
from src.losses.cosface import CosFaceLoss
from src.models.resnet50 import ResNet50
from src.models.transformer import Transformer
from src.models.partfVit import PartFVitWithLandmark


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


def test_resnet50(ResNet50):
    BATCH_SIZE = 32
    net = ResNet50(img_channels=3, num_classes=1000)
    x = torch.randn(BATCH_SIZE, 3, 112, 112)
    y = net(x)
    assert y.size() == torch.Size([BATCH_SIZE, 1000])


def test_transformer(Transformer):
    BATCH_SIZE = 32
    transformer_model = Transformer(
        feat_dim=768, mlp_dim=2048, num_heads=12, num_layers=12, dropout=0.1
    )
    x = torch.randn(BATCH_SIZE, 768)
    y = transformer_model(x)
    assert y.size() == torch.Size([BATCH_SIZE, 768])


def test_cosface(Cosface):
    BATCH_SIZE = 32
    cosface_model = Cosface(num_classes=5, feat_dim=768, margin=0.35)
    x = torch.randn(BATCH_SIZE, 768)
    y = torch.randint(0, 5, (BATCH_SIZE,)).view(BATCH_SIZE, -1)
    output = cosface_model(x, y)


def test_partfVit(PartFVitWithLandmark):
    model = PartFVitWithLandmark(
        num_identites=5,
        num_landmarks=49,
        patch_size=28,
        in_channels=3,
        image_size=112,
        feat_dim=768,
        mlp_dim=2048,
        num_heads=12,
        num_layers=6,
        dropout=0.1,
    )
    assert model(torch.rand(32, 3, 112, 112)).size() == torch.Size([32, 768])


def test_predict_model(batch_size=1):
    pass


def main():
    print("testing scripts")
    print("testing data loader")
    test_data_loader(DigiFace)
    print("testing resnet50")
    test_resnet50(ResNet50)
    print("testing transformer")
    test_transformer(Transformer)
    print("testing cosface")
    test_cosface(CosFaceLoss)
    print("testing partfVit")
    test_partfVit(PartFVitWithLandmark)
    print("testing partfViT")
    print("all tests passed!!")


if __name__ == "__main__":
    main()
