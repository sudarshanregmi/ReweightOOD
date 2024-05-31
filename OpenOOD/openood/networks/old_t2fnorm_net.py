import torch
import torch.nn as nn
import torch.nn.functional as F

class T2FNormNet(nn.Module):
    def __init__(self, backbone, tau, num_classes):
        super(T2FNormNet, self).__init__()

        self.register_buffer('tau', torch.tensor(tau))
        self.backbone = backbone
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()

        try:
            feature_size = backbone.feature_size
        except AttributeError:
            feature_size = backbone.module.feature_size

        self.new_fc = nn.Linear(feature_size, num_classes)

    def forward(self, x, return_feature=False, ood_inference=False):
        features = self.backbone(x)

        if not ood_inference:
            features = F.normalize(features, dim=-1)
        features = features / self.tau.item()

        logits_cls = self.new_fc(features)
        if return_feature:
            return logits_cls, features
        else:
            return logits_cls

    def intermediate_forward(self, x):
        features = self.backbone.intermediate_forward(x) / self.tau.item()
        return features
