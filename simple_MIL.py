import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Flatten(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        input_shape = x.shape
        output_shape = [input_shape[i] for i in range(self.dim)] + [-1]
        return x.view(*output_shape)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SimpleMIL(nn.Module):

    def __init__(
                self,
                num_instances=100,
                num_classes=1,
                pretrained=True):

        super().__init__()
        self.L = 512  # to ma z resneta wychodzić AMP2d
        self.D = 256  # att inner dim
        self.K = 1    # minimum patchy?
        self.num_instances = num_instances

        self.feature_extractor = models.resnet18(pretrained=pretrained)
        self.num_features = self.feature_extractor.fc.in_features  # selfdod
        self.feature_extractor.fc = Identity()
        # self.feature_extractor.avgpool = Identity()
        self.patch_extractor = nn.Sequential(  # nn.AdaptiveMaxPool2d(1),
                                             nn.Linear(self.num_features,
                                                       self.L),
                                             nn.ReLU())  # added

        self.attention = nn.Sequential(nn.Linear(self.L, self.D),
                                       nn.Tanh(),
                                       nn.Linear(self.D, self.K))  # added

        self.classifier = nn.Sequential(
                                        nn.Linear(self.L * self.K,
                                                  num_classes))  # added

    def forward(self, x):
        # x: bs x N x C x W x W
        bs, _, ch, w, h = x.shape
        x = x.view(bs*self.num_instances, ch, w, h)  # x: N bs x C x W x W

        H = self.feature_extractor(x)  # x: N bs x C' x W' x W'
        H = self.patch_extractor(H)  # added ~dim Nbs

        A = self.attention(H)  # added ~dim NbsxK
        A = torch.transpose(A, 1, 0)  # added ~dim KxN 10
        A = F.softmax(A, dim=1)  # ~ensure weights sum up  to unity ~dim KxN

        z = torch.mm(A, H)  # added ~dim KxN ~attention pooling
        Y = self.classifier(z)  # added
        return Y, A
