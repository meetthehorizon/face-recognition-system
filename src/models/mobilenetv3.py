import math
import torch
import torch.nn as nn

from torchvision.models.mobilenetv3 import (
    mobilenet_v3_small,
    mobilenet_v3_large,
    MobileNetV3,
)


class CustomMobileNetV3(nn.Module):
    def __init__(
        self,
        model_name: str = "small",
        image_size: int = 112,
        num_landmarks: int = 49,
    ) -> MobileNetV3:
        super(CustomMobileNetV3, self).__init__()

        assert math.sqrt(
            num_landmarks
        ).is_integer(), "num_landmarks should be a perfect square"
        assert model_name in [
            "small",
            "large",
        ], "model_name should be 'small' or 'large'"

        if model_name == "small":
            self.landmark_CNN = mobilenet_v3_small()
        elif model_name == "large":
            self.landmark_CNN = mobilenet_v3_large()

        self.num_landmarks = num_landmarks
        self.image_size = image_size

        self.classifier = nn.Linear(1000, 2 * num_landmarks)
        self._init_weights()

    def _init_weights(self):
        """initial landmarks are almost equidisant thus initial bias is changed accordingly"""
        num_per_side = int(math.sqrt(self.num_landmarks))
        temp = (
            torch.linspace(0.0, 5.0, num_per_side + 2)[1:-1]
            .unsqueeze(dim=0)
            .repeat((num_per_side, 1))
        )

        y = temp.flatten()
        x = temp.T.flatten()

        bias = torch.empty(2 * self.num_landmarks)
        bias[0::2], bias[1::2] = x, y

        self.classifier.weight.data.normal_(mean=0.0, std=1.0)
        self.classifier.bias.data.copy_(bias)

    def forward(self, x):
        out = self.landmark_CNN(x)
        landmarks = self.classifier(out)
        return landmarks


if __name__ == "__main__":
    BATCH_SIZE = 32
    NUM_CHANNELS = 3
    NUM_LANDMARKS = 49
    IMAGE_SIZE = 112
    MODEL_TYPE = "large"

    model = CustomMobileNetV3(MODEL_TYPE, IMAGE_SIZE, NUM_LANDMARKS)

    x = torch.rand(BATCH_SIZE, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    out = model(x)  # shape: (BATCH_SIZE, 2 * NUM_LANDMARKS)

    print(out.shape)
