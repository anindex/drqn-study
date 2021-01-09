from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.envs.gym import GymEnv
from src.envs.atari import AtariEnv
from src.models.dqn_fc import DQNFCModel
from src.models.dqn_cnn import DQNCNNModel
from src.models.drqn_fc import DRQNFCModel
from src.models.drqn_cnn import DRQNCNNModel
from src.replay.episodic import EpisodicMemory
from src.replay.random import RandomMemory
from src.agents.dqn import DQNAgent
from src.agents.drqn import DRQNAgent

EnvDict = {"gym":         GymEnv,
           "atari":       AtariEnv}

ModelDict = {"dqn_cnn":   DQNCNNModel,
             "dqn_fc":    DQNFCModel,
             "drqn_fc":   DRQNFCModel,
             "drqn_cnn":  DRQNCNNModel}

MemoryDict = {"episodic": EpisodicMemory,
              "random":   RandomMemory}

AgentDict = {"dqn":       DQNAgent,
             "drqn":      DRQNAgent}
