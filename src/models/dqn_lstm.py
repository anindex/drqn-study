from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.models.base import Model


class DQNLSTMModel(Model):
    def __init__(self, **kwargs):
        super(DQNLSTMModel, self).__init__(**kwargs)
        # build model
        self.lstm = nn.LSTM(self.input_dims['seq_len'] * np.prod(self.input_dims['state_shape']),
                            hidden_size=self.hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        if self.enable_dueling:  # [0]: V(s); [1,:]: A(s, a)
            self.fc2 = nn.Linear(self.hidden_dim, self.output_dims + 1)
            self.v_ind = torch.LongTensor(self.output_dims).fill_(0).unsqueeze(0)
            self.a_ind = torch.LongTensor(np.arange(1, self.output_dims + 1)).unsqueeze(0)
        else:  # one q value output for each action
            self.fc2 = nn.Linear(self.hidden_dim, self.output_dims)
        self._reset()

    def forward(self, x):
        x = x.view(x.size(0), self.input_dims[0] * self.input_dims[1])
        x = F.relu((self.fc1(x)))
        if self.enable_dueling:
            x = self.fc2(x.view(x.size(0), -1))
            v_ind_vb = Variable(self.v_ind)
            a_ind_vb = Variable(self.a_ind)
            if self.use_cuda:
                v_ind_vb = v_ind_vb.cuda()
                a_ind_vb = a_ind_vb.cuda()
            v = x.gather(1, v_ind_vb.expand(x.size(0), self.output_dims))
            a = x.gather(1, a_ind_vb.expand(x.size(0), self.output_dims))
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
            del v_ind_vb, a_ind_vb, v, a
            return x
        else:
            return self.fc2(x.view(x.size(0), -1))
