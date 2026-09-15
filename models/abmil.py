import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mspn import ABMILHead


class ABMIL(nn.Module):
    def __init__(self, in_channels=768, n_classes=2, mil_hidden_1=512, mil_hidden_2=128, attn_branches=1):
        super(ABMIL, self).__init__()
        self.abmil_head = ABMILHead(in_channels=in_channels, mil_hidden_1=mil_hidden_1, mil_hidden_2=mil_hidden_2, attn_branches=attn_branches)
        self.classifier = nn.Sequential(
            nn.Linear(mil_hidden_1*attn_branches, n_classes),
        )
    
    def forward(self, x, vis_heatmap=False):
        # if not vis_heatmap:
        x = x.squeeze(0)
        attn_feats, A_raw = self.abmil_head(x)
        logits = self.classifier(attn_feats)

        if vis_heatmap:
            return A_raw, logits
        else:
            # if self.surv:
                # hazards = torch.sigmoid(logits)
                # S = torch.cumprod(1 - hazards, dim=1)
                # return hazards, S, None # match output dim
            # else:
            return logits, None

class ABMILMSCat(nn.Module):
    def __init__(self, in_channels=768, n_classes=2, mil_hidden_1=512, mil_hidden_2=64, attn_branches=1):
        super(ABMILMSCat, self).__init__()
        self.abmil_head_5x = ABMILHead(in_channels=in_channels, mil_hidden_1=mil_hidden_1, mil_hidden_2=mil_hidden_2, attn_branches=attn_branches)
        self.abmil_head_10x = ABMILHead(in_channels=in_channels, mil_hidden_1=mil_hidden_1, mil_hidden_2=mil_hidden_2, attn_branches=attn_branches)
        self.abmil_head_20x = ABMILHead(in_channels=in_channels, mil_hidden_1=mil_hidden_1, mil_hidden_2=mil_hidden_2, attn_branches=attn_branches)
        self.ms_encoder = nn.Sequential(
            nn.Linear(mil_hidden_1, mil_hidden_2),
        )
        self.cs_attn = nn.Sequential(
            nn.Conv2d(mil_hidden_2, int(mil_hidden_2/2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(mil_hidden_2/2), 1, kernel_size=3, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(int(mil_hidden_2*3), n_classes),
        )
    
    def forward(self, x, vis_heatmap=False):
        # if not vis_heatmap:
        # x = x.squeeze(0)
        x_5x, x_10x, x_20x = x
        x_5x = x_5x.squeeze(0)
        x_10x = x_10x.squeeze(0)
        x_20x = x_20x.squeeze(0)

        attn_feats_5x, A_raw_5x = self.abmil_head_5x(x_5x)
        attn_feats_10x, A_raw_10x = self.abmil_head_10x(x_10x)
        attn_feats_20x, A_raw_20x = self.abmil_head_20x(x_20x)

        ms_feats = torch.concat((self.ms_encoder(attn_feats_5x), self.ms_encoder(attn_feats_10x), self.ms_encoder(attn_feats_20x)), dim=1)


        logits = self.classifier(ms_feats)

        if vis_heatmap:
            return (A_raw_5x, A_raw_10x, A_raw_20x), logits
        else:
            return logits, None

class ABMILMS(nn.Module):
    def __init__(self, in_channels=768, n_classes=2, mil_hidden_1=512, mil_hidden_2=64, attn_branches=1):
        super(ABMILMS, self).__init__()
        self.abmil_head_5x = ABMILHead(in_channels=in_channels, mil_hidden_1=mil_hidden_1, mil_hidden_2=mil_hidden_2, attn_branches=attn_branches)
        self.abmil_head_10x = ABMILHead(in_channels=in_channels, mil_hidden_1=mil_hidden_1, mil_hidden_2=mil_hidden_2, attn_branches=attn_branches)
        self.abmil_head_20x = ABMILHead(in_channels=in_channels, mil_hidden_1=mil_hidden_1, mil_hidden_2=mil_hidden_2, attn_branches=attn_branches)
        self.ms_encoder = nn.Sequential(
            nn.Linear(mil_hidden_1, mil_hidden_2),
        )
        self.cs_attn = nn.Sequential(
            nn.Conv2d(mil_hidden_2, int(mil_hidden_2/2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(mil_hidden_2/2), 1, kernel_size=3, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(mil_hidden_2, n_classes),
        )
    
    def forward(self, x, vis_heatmap=False):
        # if not vis_heatmap:
        # x = x.squeeze(0)
        x_5x, x_10x, x_20x = x
        x_5x = x_5x.squeeze(0)
        x_10x = x_10x.squeeze(0)
        x_20x = x_20x.squeeze(0)

        attn_feats_5x, A_raw_5x = self.abmil_head_5x(x_5x)
        attn_feats_10x, A_raw_10x = self.abmil_head_10x(x_10x)
        attn_feats_20x, A_raw_20x = self.abmil_head_20x(x_20x)

        ms_feats = torch.concat((self.ms_encoder(attn_feats_5x), self.ms_encoder(attn_feats_10x), self.ms_encoder(attn_feats_20x)), dim=0).T
        # print(ms_feats.shape)
        cs_attn = self.cs_attn(ms_feats.view(1, ms_feats.shape[0], ms_feats.shape[1], 1))
        # print(cs_attn.shape)
        cs_attn = F.softmax(cs_attn.view(1, 3), dim=1)
        cs_attn_feats = torch.mm(cs_attn, ms_feats.T)

        logits = self.classifier(cs_attn_feats)

        if vis_heatmap:
            return (A_raw_5x, A_raw_10x, A_raw_20x), logits
        else:
            return logits, None
