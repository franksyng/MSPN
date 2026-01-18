import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return A, x


class PorpoiseAMIL(nn.Module):
    def __init__(self, in_dim=768, size_arg='small', n_classes=2):
        super(PorpoiseAMIL, self).__init__()
        self.size_dict = {"small": [in_dim, 512, 256], "big": [in_dim, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=0.25, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        self.classifier = nn.Linear(size[1], n_classes)
        initialize_weights(self)

    def relocate(self, device):
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to(device)
        else:
            self.attention_net = self.attention_net.to(device)

        self.classifier = self.classifier.to(device)

    def forward(self, x, vis_heatmap=False):
        if not vis_heatmap:
            x = x.squeeze(dim=0)

        A, h = self.attention_net(x)
        A = torch.transpose(A, 1, 0)

        # if attn_only:
        #     return A, h

        A_raw = A
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        h = self.classifier(M)
        if vis_heatmap:
            return A_raw, h
        else:
            return h

    def get_slide_features(self, x, attn_only=False):
        h = x.squeeze(dim=0)

        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)

        if attn_only:
            return A

        A_raw = A
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        return M


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

