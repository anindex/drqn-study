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
        self.conv1 = nn.Conv2d(self.input_dims['seq_len'], self.kernel_num, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(self.kernel_num, self.kernel_num, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(self.kernel_num, self.kernel_num, kernel_size=3, stride=2, padding=1)
        self.lstm = nn.LSTM(self.kernel_num * 5 * 5, hidden_size=self.hidden_dim, num_layers=self.num_lstm_layer, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, self.output_dims)
        self._reset()
        self.print_model()

    def forward(self, x, hidden):
        x = x.view(x.size(0) * self.input_dims['seq_len'], *self.input_dims['state_shape'])  # batch_size * timestep, C, H, W
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), 1, -1)
        x, new_hidden = self.lstm(x, hidden)
        x = self.fc1(x)
        return x, new_hidden

    def print_model(self):
        self.logger.info('<-----------------------------------> DRQN: %s' % self.name)
        self.logger.info(self)
