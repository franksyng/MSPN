import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.ReLU()]

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



def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            # nn.init.xavier_normal_(m.weight)
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    
class AttnBlock(nn.Module):
    def __init__(self, in_dim=1024, emb_dim=256, dropout=None, n_classes=1):
        super(AttnBlock, self).__init__()
        self.attention_a = [nn.Linear(in_dim, emb_dim),
                            nn.ReLU()]

        self.attention_b = [nn.Linear(in_dim, emb_dim),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Sequential(nn.Linear(emb_dim, n_classes))

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A


class PorpoiseAMIL(nn.Module):
    def __init__(self, in_dim=512, n_classes=2):
        super(PorpoiseAMIL, self).__init__()
        if in_dim > 1024:
            size = [in_dim, 1024, 512]
        else:
            size = [in_dim, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        self.fc = nn.Sequential(*fc)
        self.attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=0.25, n_classes=1)
        self.classifier = nn.Linear(size[1], n_classes)
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

    def forward(self, x, vis_heatmap=False):
        if not vis_heatmap:
            x = x.squeeze(dim=0)
        x_ = self.fc(x)
        A, h = self.attention_net(x_)
        A = torch.transpose(A, 1, 0)

        A_raw = A
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        h = self.classifier(M)
        if vis_heatmap:
            return A_raw, h
        else:
            return h


class AMILBlock(nn.Module):
    def __init__(self, in_dim=512, emb_dim=256):
        super(AMILBlock, self).__init__()
        self.attention_net = AttnBlock(in_dim=in_dim, emb_dim=emb_dim, dropout=0.25, n_classes=1)
        initialize_weights(self)

    def relocate(self, device):
        print('[model] Num. of GPU:', torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to(device)
        else:
            self.attention_net = self.attention_net.to(device)

    def forward(self, x, attn_map=None, factor=None):
        if attn_map != None:
            scale_factor = torch.mean(factor)
            A_mask = torch.where(attn_map >= scale_factor, 1, 0)
            # print(A_mask.shape)
            mask_full = A_mask.repeat(1, x.shape[1])
            # print(x.shape, mask_full.shape)
            x = (x * mask_full) + x

        A = self.attention_net(x)
        # A = torch.transpose(A, 1, 0)

        A_raw = A
        A = F.softmax(A, dim=0)
        # M = torch.mm(A, h)
        return A_raw, A, x


class AFAMILDev(nn.Module):
    def __init__(self, in_dim=1535, emb_dim=256, block_num=2, block_type='attn', n_classes=2):
        super(AFAMILDev, self).__init__()
        if in_dim > 1024:
            size = [in_dim, 1024, 512]
        else:
            size = [in_dim, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        self.fc = nn.Sequential(*fc)

        # build AMIL blocks
        self.factor = nn.Parameter(torch.sigmoid(torch.randn(128)))
        self.amil_blocks = []
        for i in range(0, block_num):
            if block_type == 'attn':
                block = AMILBlock(size[1], emb_dim)
            else:
                print('Unknown block type.')
                raise NotImplementedError
            self.amil_blocks.append(block)
        self.classifier = nn.Sequential(nn.Linear(size[1], n_classes))
        initialize_weights(self)

    def relocate(self, device):
        print('[model] Num. of GPU:', torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.fc = nn.DataParallel(self.fc, device_ids=device_ids).to(device)
            self.amil_blocks = [nn.DataParallel(each, device_ids=device_ids).to(device) for each in self.amil_blocks]
        else:
            self.fc = self.fc.to(device)
            self.amil_blocks = [each.to(device) for each in self.amil_blocks]
        self.classifier = self.classifier.to(device)
    
    def forward(self, x, vis_heatmap=False):
        if not vis_heatmap:
            x = x.squeeze(dim=0)
        # MLP
        h = self.fc(x)
        x_ = h
        # AutoFocus
        A = None
        for block in self.amil_blocks:
            A_raw, A, h = block(h, attn_map=A, factor=self.factor)
        # print(A.shape, x_.shape)
        A = torch.transpose(A, 1, 0)
        M = torch.mm(A, x_)
        # print(M.shape)
        out = self.classifier(M)
        
        if vis_heatmap:
            return A_raw, out
        else:   
            return out

