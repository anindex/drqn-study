from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import Model


class DQNFCModel(Model):
    def __init__(self, **kwargs):
        super(DQNFCModel, self).__init__(**kwargs)
        # build model
        self.fc1 = nn.Linear(np.prod(self.input_dims['state_shape']), self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        if self.enable_dueling:  # [0]: V(s); [1,:]: A(s, a)
            self.fc3 = nn.Linear(self.hidden_dim, self.output_dims + 1)
            self.v_idx = torch.LongTensor(self.output_dims).fill_(0).unsqueeze(0)
            self.a_idx = torch.LongTensor(np.arange(1, self.output_dims + 1)).unsqueeze(0)
        else:  # one q value output for each action
            self.fc3 = nn.Linear(self.hidden_dim, self.output_dims)
        self._reset()
        self.print_model()

    def forward(self, x):
        x = x.view(x.size(0), np.prod(self.input_dims['state_shape']))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.enable_dueling:
            v = x.gather(1, self.v_idx.expand(x.size(0), self.output_dims))
            a = x.gather(1, self.a_idx.expand(x.size(0), self.output_dims))
            # now calculate Q(s, a)
            if self.dueling_type == 'avg':      # Q(s,a)=V(s)+(A(s,a)-avg_a(A(s,a)))
                # x = v + (a - a.mean(1)).expand(x.size(0), self.output_dims)   # 0.1.12
                x = v + (a - a.mean(1, keepdim=True))                           # 0.2.0
            elif self.dueling_type == 'max':    # Q(s,a)=V(s)+(A(s,a)-max_a(A(s,a)))
                # x = v + (a - a.max(1)[0]).expand(x.size(0), self.output_dims) # 0.1.12
                x = v + (a - a.max(1, keepdim=True)[0])                         # 0.2.0
            elif self.dueling_type == 'naive':  # Q(s,a)=V(s)+ A(s,a)
                x = v + a
            else:
                raise ValueError('dueling_type must be one of {\'avg\', \'max\', \'naive\'}')
        return x

    def print_model(self):
        self.logger.info('<-----------------------------------> DQNFC: %s' % self.name)
        self.logger.info(self)
