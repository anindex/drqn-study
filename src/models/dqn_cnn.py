from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.base import Model


class DQNCNNModel(Model):
    def __init__(self, **kwargs):
        super(DQNCNNModel, self).__init__(**kwargs)
        self.conv1 = nn.Conv2d(self.input_dims['seq_len'], 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.fc4 = nn.Linear(32 * 5 * 5, self.hidden_dim)
        if self.enable_dueling:  # [0]: V(s); [1,:]: A(s, a)
            self.fc5 = nn.Linear(self.hidden_dim, self.output_dims + 1)
            self.v_idx = torch.LongTensor(self.output_dims).fill_(0).unsqueeze(0)
            self.a_idx = torch.LongTensor(np.arange(1, self.output_dims + 1)).unsqueeze(0)
        else:  # one q value output for each action
            self.fc5 = nn.Linear(self.hidden_dim, self.output_dims)
        self._reset()

    def forward(self, x):
        x = x.view(x.size(0), self.input_dims[0], self.input_dims[1], self.input_dims[1])
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = F.relu((self.fc4(x.view(x.size(0), -1))))
        if self.enable_dueling:
            x = self.fc5(x)
            v_idx_vb = Variable(self.v_idx)
            a_idx_vb = Variable(self.a_idx)
            if self.use_cuda:
                v_idx_vb = v_idx_vb.cuda()
                a_idx_vb = a_idx_vb.cuda()
            v = x.gather(1, v_idx_vb.expand(x.size(0), self.output_dims))
            a = x.gather(1, a_idx_vb.expand(x.size(0), self.output_dims))
            # now calculate Q(s, a)
            if self.dueling_type == 'avg':      # Q(s,a)=V(s)+(A(s,a)-avg_a(A(s,a)))
                x = v + (a - a.mean(1).expand(x.size(0), self.output_dims))
            elif self.dueling_type == 'max':    # Q(s,a)=V(s)+(A(s,a)-max_a(A(s,a)))
                x = v + (a - a.max(1)[0].expand(x.size(0), self.output_dims))
            elif self.dueling_type == 'naive':  # Q(s,a)=V(s)+ A(s,a)
                x = v + a
            else:
                raise ValueError('dueling_type must be one of {\'avg\', \'max\', \'naive\'}')
            del v_idx_vb, a_idx_vb, v, a
            return x
        else:
            return self.fc5(x.view(x.size(0), -1))
