import torch
from src.models.transformer import Transformer


def test():
    model = Transformer(
        num_landmarks=49,
        patch_size=28,
        in_channels=3,
        image_size=112,
        feat_dim=768,
        mlp_dim=2048,
        num_heads=11,
        num_layers=12,
        dropout=0.0,
    )
    batch = torch.randn(1, 3, 112, 112)
    output = model(batch)
    assert output.shape == (1, 49, 2)


if __name__ == "__main__":
    test()
    print("Test passed!")
