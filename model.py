import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Define the Regression Model
class RegressionModel(nn.Module):
    def __init__(self, num_classes=1):
        super(RegressionModel, self).__init__()
        self.resnet = models.resnet50()
        # Modify the first layer to accept 1 channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Mish(),
            nn.Linear(256, 64),
            nn.Mish(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        features = self.resnet(x)
        output = self.linear_layers(features)
        return output
    

     