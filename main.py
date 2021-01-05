import yaml
import argparse
import logging
from os.path import join, dirname, abspath

from src.utils.factory import EnvDict, ModelDict, MemoryDict, AgentDict

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python main.py cartpole_dqn.yaml')
parser.add_argument('config_file', help='configuration to run experiment', type=str)
args = parser.parse_args()

ROOT_DIR = dirname(abspath(__file__))
config_file = join(ROOT_DIR, 'configs', args.config_file)
with open(config_file, 'r') as f:
    params = yaml.full_load(f)
params['model_file'] = join(ROOT_DIR, 'saves', params['model_file'])
params['env']['root_dir'] = ROOT_DIR

env_prototype = EnvDict[params['env_type']]
model_prototype = ModelDict[params['model_type']]
memory_prototype = MemoryDict[params['memory_type']]
agent = AgentDict[params['agent_type']](env_prototype=env_prototype,
                                        model_prototype=model_prototype,
                                        memory_prototype=memory_prototype,
                                        **params)
if params['mode'] == 'train':
    agent.fit_model()
elif params['mode'] == 'test':
    agent.test_model()
