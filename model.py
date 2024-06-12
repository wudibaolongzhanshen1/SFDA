import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class feature_extractor(nn.Module):
    def __init__(self,weights=ResNet50_Weights):
        super(feature_extractor, self).__init__()
        if weights:
            model = models.resnet50(weights=weights)
        else:
            model = models.resnet50(weights=None)
        self.model = model
        self.feature_extractor = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return x

class classifier_s2t(nn.Module):
    def __init__(self,num_classes=10):
        super(classifier_s2t, self).__init__()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class classifier_t(nn.Module):
    def __init__(self,num_classes=10):
        super(classifier_t, self).__init__()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class MyModel(nn.Module):
    def __init__(self,weights=ResNet50_Weights,num_classes=10):
        super(MyModel, self).__init__()
        self.feature_extractor = feature_extractor(weights=weights)
        self.classifier_s2t = classifier_s2t(num_classes)
        self.classifier_t = classifier_t(num_classes)
    def forward(self, x):
        x = self.feature_extractor(x)
        x1 = self.classifier_s2t(x)
        x2 = self.classifier_t(x)
        return x1,x2

class resnet50(nn.Module):
    def __init__(self,weights=ResNet50_Weights,num_classes=10):
        super(resnet50, self).__init__()
        self.feature_extractor = feature_extractor(weights=weights)
        self.classifier = classifier_t(num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x