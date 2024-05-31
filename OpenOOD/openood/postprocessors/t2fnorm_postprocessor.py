from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor


class T2FNormPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(T2FNormPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.percentile = self.args.percentile
        self.tau = self.args.tau
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net.forward_threshold(data, self.percentile, self.tau)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        output = net(data)
        _, pred = torch.max(output, dim=1)
        return pred, energyconf

    def set_hyperparam(self, hyperparam: list):
        self.percentile = hyperparam[0]
        self.tau = hyperparam[1]

    def get_hyperparam(self):
        return [self.percentile, self.tau]
