import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from batch_idx_sel import batched_index_select
from nystrom_attention import NystromAttention


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
        # device = x.device

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
        A = F.softmax(A, dim=1)
        # A = F.softmax(A / torch.sqrt(torch.tensor(
        #     Q.shape[2], dtype=torch.float32, device=device)), dim=1)

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


# ------------------------MA-MIL------------------------------------
class MultiAttentionMIL(nn.Module):
    def __init__(self, num_classes, pretrained,
                 use_dropout=False, n_dropout=0.4):

        super(MultiAttentionMIL, self).__init__()
        self.num_classes = num_classes
        self.use_dropout = use_dropout
        self.n_dropout = n_dropout

        self.D = 128

        self.feature_extractor = models.resnet18(pretrained=pretrained)
        self.num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = Identity()

        self.fc1 = nn.Sequential(
            nn.Linear(self.num_features, self.D),
            nn.ReLU(),
        )
        self.attention1 = nn.Sequential(
            nn.Linear(self.D, self.D), nn.Tanh(), nn.Linear(self.D, 1)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.ReLU(),
        )
        self.attention2 = nn.Sequential(
            nn.Linear(self.D, self.D), nn.Tanh(), nn.Linear(self.D, 1)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.ReLU(),
        )
        self.attention3 = nn.Sequential(
            nn.Linear(self.D, self.D), nn.Tanh(), nn.Linear(self.D, 1)
        )

        self.fc4 = nn.Sequential(nn.Linear(self.D, self.num_classes))

    def forward(self, x):
        ################
        x1 = x.squeeze(0)
        x1 = self.feature_extractor(x1)
        x1 = self.fc1(x1)
        if self.use_dropout:
            x1 = nn.Dropout(self.n_dropout)(x1)
        # -------
        a1 = self.attention1(x1)
        a1 = torch.transpose(a1, 1, 0)
        a1 = nn.Softmax(dim=1)(a1)
        # -------
        m1 = torch.mm(a1, x1)
        m1 = m1.view(-1, 1 * self.D)

        ################
        x2 = self.fc2(x1)
        if self.use_dropout:
            x2 = nn.Dropout(self.n_dropout)(x2)
        # -------
        a2 = self.attention2(x2)
        a2 = torch.transpose(a2, 1, 0)
        a2 = nn.Softmax(dim=1)(a2)
        # -------
        m2 = torch.mm(a2, x2)
        m2 = m2.view(-1, 1 * self.D)
        m2 += m1

        ################
        x3 = self.fc3(x2)
        if self.use_dropout:
            x3 = nn.Dropout(self.n_dropout)(x3)
        # -------
        a3 = self.attention3(x3)
        a3 = torch.transpose(a3, 1, 0)
        a3 = nn.Softmax(dim=1)(a3)
        # -------
        m3 = torch.mm(a3, x3)
        m3 = m3.view(-1, 1 * self.D)
        m3 += m2

        result = self.fc4(m3)

        return result, a1, a2, a3
# -----------------------------------------------------------------


# ------------------------GMA-MIL------------------------------------
class GatedMultiAttentionMIL(nn.Module):
    def __init__(self, num_classes, pretrained,
                 use_dropout=False, n_dropout=0.4):

        super(GatedMultiAttentionMIL, self).__init__()
        self.num_classes = num_classes
        self.use_dropout = use_dropout
        self.n_dropout = n_dropout

        self.D = 128

        self.feature_extractor = models.resnet18(pretrained=pretrained)
        self.num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = Identity()

        self.fc1 = nn.Sequential(
            nn.Linear(self.num_features, self.D),
            nn.ReLU()
        )
        self.attention_V1 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.Tanh()
        )
        self.attention_U1 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.Sigmoid()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.ReLU()
        )
        self.attention_V2 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.Tanh(),
        )
        self.attention_U2 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.Sigmoid()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.ReLU(),
        )
        self.attention_V3 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.Tanh(),
        )
        self.attention_U3 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.Sigmoid()
        )

        self.attention_weights1 = nn.Linear(self.D, 1)
        self.attention_weights2 = nn.Linear(self.D, 1)
        self.attention_weights3 = nn.Linear(self.D, 1)

        self.fc4 = nn.Sequential(nn.Linear(self.D, self.num_classes))

    def forward(self, x):
        ################
        x1 = x.squeeze(0)
        x1 = self.feature_extractor(x1)
        x1 = self.fc1(x1)
        if self.use_dropout:
            x1 = nn.Dropout(self.n_dropout)(x1)
        # -------
        A_V1 = self.attention_V1(x1)  # N x D
        A_U1 = self.attention_U1(x1)  # N x D
        A1 = self.attention_weights1(torch.mul(A_V1, A_U1))
        A1 = torch.transpose(A1, 1, 0)
        A1 = F.softmax(A1, dim=1)  # ~ensure weights sum up  to unity
        m1 = torch.mm(A1, x1)  # ~attention pooling
        m1 = m1.view(-1, 1 * self.D)

        ################
        x2 = self.fc2(x1)
        if self.use_dropout:
            x2 = nn.Dropout(self.n_dropout)(x2)
        # -------
        A_V2 = self.attention_V2(x1)  # N x D
        A_U2 = self.attention_U2(x1)  # N x D
        A2 = self.attention_weights2(torch.mul(A_V2, A_U2))
        A2 = torch.transpose(A2, 1, 0)
        A2 = F.softmax(A2, dim=1)  # ~ensure weights sum up  to unity
        m2 = torch.mm(A2, x1)  # ~attention pooling
        m2 = m2.view(-1, 1 * self.D)
        m2 += m1
        ################
        x3 = self.fc3(x2)
        if self.use_dropout:
            x3 = nn.Dropout(self.n_dropout)(x3)
        # -------
        A_V3 = self.attention_V3(x1)  # N x D
        A_U3 = self.attention_U3(x1)  # N x D
        A3 = self.attention_weights3(torch.mul(A_V3, A_U3))
        A3 = torch.transpose(A3, 1, 0)
        A3 = F.softmax(A3, dim=1)  # ~ensure weights sum up  to unity
        m3 = torch.mm(A3, x1)  # ~attention pooling
        m3 = m3.view(-1, 1 * self.D)
        m3 += m2

        result = self.fc4(m3)

        return result, A1, A2, A3
# -----------------------------------------------------------------


# ---------------------------TRANS MIL-----------------------------
class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
                                    dim=dim,
                                    dim_head=dim//8,
                                    heads=8,
                                    num_landmarks=dim//2,
                                    pinv_iterations=6,
                                    residual=True,
                                    dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)\
            + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, num_classes,
                 pretrained=True):

        super(TransMIL, self).__init__()

        self.feature_extractor = models.resnet18(pretrained=pretrained)
        self.num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = Identity()

        self.pos_layer = PPEG(dim=512)
        # self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = num_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    def forward(self, x):

        bs, num_instances, _, _, _ = x.shape
        my_device = x.device

        x1 = x.squeeze(0)  # [B*n, ch, h, w]
        x1 = self.feature_extractor(x1)  # [B*n, 512]

        # h = self.fc(x1.view(bs, num_instances, 512))  # [B, n, 512]
        h = x1.view(bs, num_instances, 512)
        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens.to(my_device)
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        logits = self._fc2(h)  # [B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)

        return logits, Y_hat, Y_prob
