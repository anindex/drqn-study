from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import Model


class DRQNCNNModel(Model):
    def __init__(self, **kwargs):
        super(DRQNCNNModel, self).__init__(**kwargs)
        # build model
        self.kernel_num = kwargs.get('kernel_num', 32)
        self.num_lstm_layer = kwargs.get('num_lstm_layer', 1)
        self.conv1 = nn.Conv2d(self.input_dims['state_shape'][0], self.kernel_num, kernel_size=7, stride=2)
        self.conv2 = nn.Conv2d(self.kernel_num, self.kernel_num, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(self.kernel_num, self.kernel_num, kernel_size=3, stride=1)
        self.lstm = nn.LSTM(self.kernel_num * 16 * 16, hidden_size=self.hidden_dim, num_layers=self.num_lstm_layer, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, self.output_dims)
        self._reset()
        self.print_model()

    def forward(self, x, hidden):
        batch_size, seq_len, stack_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, -1, *self.input_dims['state_shape'][1:])  # batch_size * seq_len, C * stack_len, H, W
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, seq_len, -1)
        x, new_hidden = self.lstm(x, hidden)
        new_hidden = tuple(h.detach() for h in new_hidden)  # detach to avoid inplace modification error in computing grad
        x = self.fc1(x)
        return x, new_hidden

    def print_model(self):
        self.logger.info('<-----------------------------------> DRQN: %s' % self.name)
        self.logger.info(self)
