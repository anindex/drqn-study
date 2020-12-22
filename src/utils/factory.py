from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from envs.gym import GymEnv
from envs.atari import AtariEnv
from models.dqn_fc import DQNFCModel
from models.dqn_cnn import DQNCNNModel
from models.dqn_lstm import DQNLSTMModel
from replay.episodic import EpisodicMemory
from replay.random import RandomMemory
from agents.dqn import DQNAgent

EnvDict = {"gym":         GymEnv,
           "atari":       AtariEnv}

ModelDict = {"dqn-cnn":   DQNCNNModel,
             "dqn_fc":    DQNFCModel,
             "dqn_lstm":  DQNLSTMModel}

MemoryDict = {"episodic": EpisodicMemory,
              "random":   RandomMemory}

AgentDict = {"dqn":   DQNAgent}
