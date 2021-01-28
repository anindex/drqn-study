from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import Model


class DQNCNNModel(Model):
    def __init__(self, **kwargs):
        super(DQNCNNModel, self).__init__(**kwargs)
        self.kernel_num = kwargs.get('kernel_num', 32)
        self.conv1 = nn.Conv2d(self.input_dims['state_shape'][0], self.kernel_num, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(self.kernel_num, self.kernel_num * 2, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(self.kernel_num * 2, self.kernel_num * 2, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(self.kernel_num * 2 * 7 * 7, self.hidden_dim)
        if self.enable_dueling:  # [0]: V(s); [1,:]: A(s, a)
            self.fc5 = nn.Linear(self.hidden_dim, self.output_dims + 1)
            self.v_idx = torch.LongTensor(self.output_dims).fill_(0).unsqueeze(0).cuda()
            self.a_idx = torch.LongTensor(np.arange(1, self.output_dims + 1)).unsqueeze(0).cuda()
        else:  # one q value output for each action
            self.fc5 = nn.Linear(self.hidden_dim, self.output_dims)
        self._reset()
        self.print_model()

    def forward(self, x):
        x = x.view(x.size(0), *self.input_dims['state_shape'])  # batch_size, C * stack_len, H, W
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        x = self.fc5(x)
        if self.enable_dueling:
            v = x.gather(1, self.v_idx.expand(x.size(0), self.output_dims))
            a = x.gather(1, self.a_idx.expand(x.size(0), self.output_dims))
            # now calculate Q(s, a)
            if self.dueling_type == 'avg':      # Q(s,a)=V(s)+(A(s,a)-avg_a(A(s,a)))
                x = v + (a - a.mean(1).expand(x.size(0), self.output_dims))
            elif self.dueling_type == 'max':    # Q(s,a)=V(s)+(A(s,a)-max_a(A(s,a)))
                x = v + (a - a.max(1)[0].expand(x.size(0), self.output_dims))
            elif self.dueling_type == 'naive':  # Q(s,a)=V(s)+ A(s,a)
                x = v + a
            else:
                raise ValueError('dueling_type must be one of {\'avg\', \'max\', \'naive\'}')
        return x

    def print_model(self):
        self.logger.info('<-----------------------------------> DQNCNN: %s' % self.name)
        self.logger.info(self)
