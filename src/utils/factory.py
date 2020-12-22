from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.envs.gym import GymEnv
from src.envs.atari import AtariEnv
from src.models.dqn_fc import DQNFCModel
from src.models.dqn_cnn import DQNCNNModel
from src.models.dqn_lstm import DQNLSTMModel
from src.replay.episodic import EpisodicMemory
from src.replay.random import RandomMemory
from src.agents.dqn import DQNAgent

EnvDict = {"gym":         GymEnv,
           "atari":       AtariEnv}

ModelDict = {"dqn-cnn":   DQNCNNModel,
             "dqn_fc":    DQNFCModel,
             "dqn_lstm":  DQNLSTMModel}

MemoryDict = {"episodic": EpisodicMemory,
              "random":   RandomMemory}

AgentDict = {"dqn":   DQNAgent}
