import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from einops import rearrange, repeat

""" In this project, we use AMIL from PORPOISE to be the baseline"""


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes (experimental usage for multiclass MIL)
"""


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            # nn.ReLU()]
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        # if self.attention_c.weight.grad != None:
            # print(self.attention_c.weight.grad.shape)
        return A, x


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class CLAM_SB(nn.Module):
    def __init__(self, in_dim=512, n_classes=2):
        super(CLAM_SB, self).__init__()
        size = [in_dim, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=0.25, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Linear(size[1], n_classes)
        initialize_weights(self)

    def relocate(self, device):
        print('[model] Num. of GPU:', torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to(device)
        else:
            self.attention_net = self.attention_net.to(device)

        self.classifier = self.classifier.to(device)

    def forward(self, h, vis_heatmap=False):
        if not vis_heatmap:
            h = h.squeeze(dim=0)
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        A_raw = A  # [1, N]
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        logits = self.classifier(M)
        if vis_heatmap:
            return A_raw, logits
        else:
            return logits

class CLAM_MB(nn.Module):
    def __init__(self, in_dim=512, n_classes=2):
        super(CLAM_MB, self).__init__()
        size = [in_dim, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=0.25, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifier = nn.ModuleList(bag_classifiers)
        self.n_classes = n_classes
        initialize_weights(self)

    def relocate(self, device):
        print('[model] Num. of GPU:', torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to(device)
        else:
            self.attention_net = self.attention_net.to(device)

        self.classifier = self.classifier.to(device)

    def forward(self, h, vis_heatmap=False):
        if not vis_heatmap:
            h = h.squeeze(dim=0)
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)

        # if attn_only:
        #     return A, h

        # print(A)
        A_raw = A
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        # print(M.shape)
        logits = torch.empty(1, self.n_classes).float().to(M.device)
        # print(logits)
        for c in range(self.n_classes):
            logits[0, c] = self.classifier[c](M[c])
            # print(logits[0, c])
        # print(logits)
        # h = self.classifier(M)
        if vis_heatmap:
            return A_raw, logits
        else:
            return logits



class PosEncAMIL(nn.Module):
    def __init__(self, in_dim=512, n_classes=2):
        super(PosEncAMIL, self).__init__()
        size = [in_dim, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        self.fc = nn.Sequential(*fc)
        self.attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=0.25, n_classes=1)
        self.classifier = nn.Linear(size[1], n_classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_dim))
        initialize_weights(self)

    def relocate(self, device):
        print('[model] Num. of GPU:', torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to(device)
            self.fc = nn.DataParallel(self.fc, device_ids=device_ids).to(device)
        else:
            self.attention_net = self.attention_net.to(device)
            self.fc = self.fc.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, h, vis_heatmap=False):
        device = h.device
        h = h.unsqueeze(0)
        # cls token
        B = h.shape[0]
        # print(self.cls_token.shape)
        cls_tokens = self.cls_token.expand(B, -1, -1).to(device)
        # print(cls_tokens.shape)
        # print(h.shape)
        h = torch.cat((cls_tokens, h), dim=1)
        h = positional_encoding(h, h.size(2), h.size(1), device=device)
        # if not vis_heatmap:
        h = h.squeeze(dim=0)
        h = self.fc(h)
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)

        A_raw = A
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        h = self.classifier(M)
        if vis_heatmap:
            return A_raw, h
        else:
            return h


def positional_encoding(x, d_model, max_len, device):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model).to(device)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    return F.dropout(x + pe[:x.size(0)], p=0.1)
