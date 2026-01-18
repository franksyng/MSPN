import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from einops import rearrange, repeat
from sklearn import preprocessing
import numpy as np
from utils.universal_utils import initialize_weights

""" In this project, we use AMIL from PORPOISE to be the baseline"""


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes (experimental usage for multiclass MIL)
"""

class MaxPool(nn.Module):
    def __init__(self, in_dim=768, n_classes=2):
        super(MaxPool, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        # self.surv = surv
        self.classifier = nn.Linear(in_dim, n_classes)
    
    def forward(self, h):
        # print(h.shape)
        h = h.squeeze(0).transpose(1, 0)
        h = self.maxpool(h)
        # print(h.shape)
        logits = self.classifier(h.transpose(1, 0))
        return logits, None

class MeanPool(nn.Module):
    def __init__(self, in_dim=768, n_classes=2):
        super(MeanPool, self).__init__()
        self.meanpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_dim, n_classes)
        # self.surv = surv
    
    def forward(self, h):
        h = h.squeeze(0).transpose(1, 0)
        h = self.meanpool(h)
        logits = self.classifier(h.transpose(1, 0))
        return logits, None


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

