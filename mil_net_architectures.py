import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from batch_idx_sel import batched_index_select


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# ----------------------------GATED MIL----------------------------
class GatedMIL(nn.Module):

    def __init__(
                self,
                num_classes=1,
                pretrained=True):

        super().__init__()
        self.L = 512  # to ma z resneta wychodzić AMP2d
        self.D = 128  # att inner dim
        self.K = 1    # minimum patchy?

        self.feature_extractor = models.resnet18(pretrained=pretrained)
        self.num_features = self.feature_extractor.fc.in_features  # selfdod
        self.feature_extractor.fc = Identity()
        # # self.feature_extractor.avgpool = Identity()
        # self.patch_extractor = nn.Sequential(  # nn.AdaptiveMaxPool2d(1),
        #                                      nn.Linear(self.num_features,
        #                                                self.L),
        #                                      nn.ReLU())

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
                                        nn.Linear(self.L * self.K,
                                                  num_classes))

    def forward(self, x):
        # x: bs x N x C x W x W

        bs, num_instances, ch, w, h = x.shape
        x = x.view(bs*num_instances, ch, w, h)  # x: N bs x C x W x W
        H = self.feature_extractor(x)  # x: N bs x C' x W' x W'
        # H = self.patch_extractor(H)  # added ~dim Nbs
        H = H.view(bs, num_instances, -1)
        A_V = self.attention_V(H)  # bs x N x D
        A_U = self.attention_U(H)  # bs x N x D
        A = self.attention_weights(torch.mul(A_V, A_U))
        A = torch.transpose(A, 2, 1)  # added ~dim bsxKxN 10
        A = F.softmax(A, dim=2)  # ~ensure weights sum up  to unity ~dim bsxKxN
        m = torch.matmul(A, H)  # added ~dim bsxKxN ~attention pooling

        Y = self.classifier(m)  # added
        # print(Y)
        return Y, A
# -----------------------------------------------------------------


# ----------------------Dual Stream MIL----------------------------
class DSMIL(nn.Module):

    def __init__(
                self,
                num_classes=1,
                pretrained=True,
                nonlinear=False):

        super().__init__()

        self.feature_extractor = models.resnet18(pretrained=pretrained)
        self.num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = Identity()

        self.IC_fc = nn.Linear(self.num_features, num_classes)

        if nonlinear:
            self.lin = nn.Sequential(nn.Linear(self.num_features,
                                               self.num_features),
                                     nn.ReLU())

            self.q = nn.Sequential(nn.Linear(self.num_features, 128),
                                   nn.Tanh())

        else:
            self.lin = nn.Identity()
            self.q = nn.Linear(self.num_features, 128)

        self.v = nn.Sequential(nn.Dropout(0.),
                               nn.Linear(self.num_features, self.num_features))

        self.fcc = nn.Conv1d(num_classes, num_classes,
                             kernel_size=self.num_features)

    def forward(self, x):
        # bs-batch, N-num_instances, K-fts_per_inst CH-channels W/H-wdth/hght
        bs, num_instances, ch, h, w = x.shape
        device = x.device

        x = x.view(bs*num_instances, ch, h, w)  # x: N bs x CH x W x W
        feats = self.feature_extractor(x)  # feats: N bs x CH' x W' x W'
        feats = feats.view(bs, num_instances, -1)  # bs x N x K
        c = self.IC_fc(feats)  # bs x N x C

        feats = self.lin(feats)
        V = self.v(feats)  # bs x N x V
        Q = self.q(feats).view(bs, num_instances, -1)  # bs x N x Q

        _, m_indices = torch.sort(c, 1, descending=True)  # bs x N x C

        # bs x C x K
        m_feats = batched_index_select(feats, dim=1, index=m_indices[:, 0, :])
        q_max = self.q(m_feats)  # bs x C x Q

        A = torch.matmul(Q, q_max.transpose(1, 2))  # bs x N x C
        A = F.softmax(A / torch.sqrt(torch.tensor(
            Q.shape[1], dtype=torch.float32, device=device)), dim=1)

        B = torch.matmul(A.transpose(1, 2), V)  # bs x C x V
        # B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V

        C = self.fcc(B)  # bs x C x 1
        C = C.view(bs, -1)  # bs x C

        Y = C

        return Y, A, B, c
# -----------------------------------------------------------------


# ---------------------------SIMPLE MIL----------------------------
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
# -----------------------------------------------------------------
