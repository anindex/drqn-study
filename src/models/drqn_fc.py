from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import Model


class DRQNFCModel(Model):
    def __init__(self, **kwargs):
        super(DRQNFCModel, self).__init__(**kwargs)
        # build model
        self.num_lstm_layer = kwargs.get('num_lstm_layer', 1)
        self.lstm = nn.LSTM(self.input_dims['seq_len'] * np.prod(self.input_dims['state_shape']),
                            hidden_size=self.hidden_dim, num_layers=self.num_lstm_layer, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dims)
        self._reset()
        self.print_model()

    def forward(self, x, hidden):
        x = x.view(x.size(0), -1, self.input_dims['seq_len'] * np.prod(self.input_dims['state_shape']))
        x, new_hidden = self.lstm(x, hidden)
        new_hidden = tuple(h.detach() for h in new_hidden)  # detach to avoid inplace modification error in computing grad
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, new_hidden

    def print_model(self):
        self.logger.info('<-----------------------------------> DRQN: %s' % self.name)
        self.logger.info(self)
