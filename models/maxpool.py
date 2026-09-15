import torch.nn as nn


class MaxPool(nn.Module):
    def __init__(self, in_dim=768, n_classes=2):
        super(MaxPool, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, h):
        h = h.squeeze(0).transpose(1, 0)
        h = self.maxpool(h)
        logits = self.classifier(h.transpose(1, 0))
        return logits, None
