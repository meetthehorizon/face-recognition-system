import torch
import torch.nn as nn
import torch.nn.functional as F


class block(nn.Module):
    """Basic block of the ResNet50 architecture that repeats itself multiple times

    Attributes
    ----------
        in_channels: int
                    number of input channels
        out_channels: int
                    number of output channels
        identity_downsample: np.ndarray
                    identity downsample layer
        stride: int
                    stride of the convolutional layer
    """

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        """
        Parameters
        ----------
            in_channels: int
                        The number of input channels
            out_channels: int
                        The number of output channels
            identity_downsample: np.ndarray
                        The identity downsample layer
            stride: int
                        The stride of the convolutional layer
        """
        super(block, self).__init__()
        self.expansion = 4  # This is the expansion factor

        # Initialising the common layers used in the network
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )  # Since we are using 1*1 kernel size the padding is 0
        self.bn1 = nn.BatchNorm2d(out_channels)  # Batch Normalization
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )  # Since we are using 3*3 kernel size the padding is 1
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )  # Here the output channels are multiplied by the expansion factor
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # Forward Function in order to pass the given input through the block
    def forward(self, x):
        """Computes the forward pass of the block

        Parameters
        ----------
            x: torch.Tensor
                The input tensor

        Returns
        -------
            torch.Tensor
                The output tensor
        """
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
    """ResNet50 architecture

    reference: https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, block, layers, img_channels, num_classes):
        super(ResNet, self).__init__()
        """
        Parameters
        ----------
            block: np.ndarray
                        The basic block that is going to be used multiple times
            layers: np.ndarray
                        The number of times the basic block is going to be used
            img_channels: int
                        The number of input channels
            num_classes: int
                        The number of output classes
        """
        # Initial layers
        self.in_channels = 64
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers
        self.layer1 = self._make_layer(
            block, layers[0], out_channels=64, stride=1
        )  # Here the output will be 64*4=256
        self.layer2 = self._make_layer(
            block, layers[1], out_channels=128, stride=2
        )  # Here the output will be 128*4=512
        self.layer3 = self._make_layer(
            block, layers[2], out_channels=256, stride=2
        )  # Here the output will be 256*4=1024
        self.layer4 = self._make_layer(
            block, layers[3], out_channels=512, stride=2
        )  # Here the output will be 512*4=2048

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Computes the forward pass of the ResNet50 architecture
        Parameters
        ----------
            x: torch.Tensor
                The input tensor
        Returns
        -------
            torch.Tensor
                The output tensor
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)

        return out

    def _make_layer(self, block, num_residual_blocks, out_channels, stride=1):
        """Creates a layer of the ResNet50 architecture
        Parameters
        ----------
            block: np.ndarray
                The basic block that is going to be used multiple times
            num_residual_blocks: int
                The number of times the basic block is going to be used
            out_channels: int
                The number of output channels
            stride: int
                The stride of the convolutional layer
        Returns
        -------
            nn.Sequential
                The layer of the ResNet50 architecture
        """
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
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)


def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channels, num_classes)


if __name__ == "__main__":
    print("passed")
