import torch
def test_resnet(ResNet50):
    BATCH_SIZE = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResNet50(img_channels=3, num_classes=1000).to(device)
    x = torch.randn(BATCH_SIZE, 3, 112, 112)
    # If the image size is aXb  then we will have to pass a/4,b/4 in forward method of the block class
    # out += torch.nn.functional.interpolate(identity, size=(a/4, b/4))
    # out = self.relu(out)

    # return out
    x = x.to(device)
    y = net(x).to(device)
    print(y.shape)
    return y.size() == torch.Size([BATCH_SIZE, 1000])
    