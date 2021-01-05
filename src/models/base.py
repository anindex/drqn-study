from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging

import torch
import torch.nn as nn


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    # out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))              # 0.1.12
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))  # 0.2.0
    return out


def weight_reset(m):
    reset_parameters = getattr(m, 'reset_parameters', None)
    if callable(reset_parameters):
        m.reset_parameters()


class Model(nn.Module):  # TODO: enable_dueling is untested
    def __init__(self, name='Model', **kwargs):
        super(Model, self).__init__()
        # logging
        self.logger = kwargs.get('logger', logging.getLogger(__name__))
        # params
        self.name = name
        self.model_file = kwargs.get('model_file', None)
        self.hidden_dim = kwargs.get('hidden_dim', 128)
        self.use_cuda = kwargs.get('use_cuda', True)
        self.dtype = kwargs.get('dtype', None)
        if self.dtype is None:
            self.dtype = torch.float32
        else:
            self.dtype = eval(self.dtype)
        # model_params
        self.enable_dueling = kwargs.get('enable_dueling', False)
        self.dueling_type = kwargs.get('dueling_type', 'avg')
        self.input_dims = {
            'seq_len': kwargs.get('seq_len', 4),
            'state_shape': kwargs.get('state_shape')
        }
        self.output_dims = kwargs.get('action_dim')

    def print_model(self):
        self.logger.info('<-----------------------------------> Model')
        self.logger.info(self)

    def _reset(self):           # NOTE: should be called at each child's __init__
        self.apply(weight_reset)
        self.type(self.dtype)   # put on gpu if possible

    def forward(self, input):
        raise NotImplementedError()
