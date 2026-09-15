import torch.nn as nn


class MeanPool(nn.Module):
    def __init__(self, in_dim=768, n_classes=2):
        super(MeanPool, self).__init__()
        self.meanpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, h):
        h = h.squeeze(0).transpose(1, 0)
        h = self.meanpool(h)
        logits = self.classifier(h.transpose(1, 0))
        return logits, None
