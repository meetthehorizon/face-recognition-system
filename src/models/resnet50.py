import torch
import torch.nn as nn
import torch.nn.functional as F


class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        # Initialising the common layers used in the network
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    # Forward Function in order to pass the given input through the block
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(x)

        out += torch.nn.functional.interpolate(identity, size=(28, 28))
        out = self.relu(out)

        return out


# Making the resnet model
class ResNet(nn.Module):
    # We are going to use the basic block like this: 1. Block used [3,4,6,3]
    # The parameters are defined as follows:
    # block: It is the basic block that is going to be used multiple times during our implementation
    # layers: The number of times we need to use the BasicBlock
    def __init__(self, block, layers, img_channels, num_classes):
        super(ResNet, self).__init__()

        # Initializing the initial layers
        # Note these are not the resnet layers but the initial layers
        self.in_channels = 64
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(
            block, layers[3], out_channels=512, stride=2
        )  # Now here the output will be 512*4=2048

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(
            out
        )  # Calls make_layer which in turn calls block multiple times
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)

        return out

    # This is the make_layer function which makes a layers according to our requirements.
    def _make_layer(self, block, num_residual_blocks, out_channels, stride=1):
        identity_downsample = None
        layers = []
        # If identity still has quarter and the out_channels has expanded
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * 4, kernel_size=1, stride=stride
                ),
                nn.BatchNorm2d(out_channels * 4),
            )
        layers.append(
            block(self.in_channels, out_channels, identity_downsample, stride)
        )
        self.in_channels = out_channels * 4
        for _ in range(num_residual_blocks - 1):
            layers.append(
                block(self.in_channels, out_channels)
            )  # in_channels is 256->64, 64*4(256) again and stride=1
            # and we don't need to downsample since in_channels==out_channels.
            # Even here the output is the same as input i.e 256
        return nn.Sequential(*layers)


def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channels, num_classes)


def test():
    BATCH_SIZE = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResNet50(img_channels=3, num_classes=1000).to(device)
    x = torch.randn(BATCH_SIZE, 3, 96, 112)
    # If the image size is aXb  then we will have to pass a/4,b/4 in forward method of the block class
    # out += torch.nn.functional.interpolate(identity, size=(a/4, b/4))
    # out = self.relu(out)

    # return out
    x = x.to(device)
    y = net(x).to(device)
    assert y.size() == torch.Size([BATCH_SIZE, 1000])
    print(y.shape)


if __name__ == "__main__":
    print("passed")
