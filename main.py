import yaml
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from os.path import join, dirname, abspath

from src.utils.factory import EnvDict, ModelDict, MemoryDict, AgentDict

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python main.py cartpole_dqn.yaml')
parser.add_argument('config_file', help='configuration to run experiment', type=str)
parser.add_argument('--run', help='number of runs', type=int, default=1)
parser.add_argument('--plot', help='specify plotting', type=bool, default=True)
args = parser.parse_args()

ROOT_DIR = dirname(abspath(__file__))
config_file = join(ROOT_DIR, 'configs', args.config_file)
with open(config_file, 'r') as f:
    params = yaml.full_load(f)

# some modifications for ease of use
now = datetime.now().strftime("%d%m%Y%H%M%S")
params['model']['model_file'] = join(ROOT_DIR, 'saves', now + '_' + params['model']['model_file'])
params['env']['root_dir'] = ROOT_DIR
log_folder_name = params['env']['game'] + '_' + params['model_type'] + '_' + params['memory_type'] + now
params['log_folder'] = join(ROOT_DIR, 'logs', log_folder_name)

env_prototype = EnvDict[params['env_type']]
model_prototype = ModelDict[params['model_type']]
memory_prototype = MemoryDict[params['memory_type']]
agent = AgentDict[params['agent_type']](env_prototype=env_prototype,
                                        model_prototype=model_prototype,
                                        memory_prototype=memory_prototype,
                                        **params)

bins = [i ** 10 for i in range(7)]
max_abs_q_log = []  # per step
tderr_log = []  # per step
total_avg_score_log = []  # per eps
run_avg_score_log = []  # per eps
step_log = None
eps_log = None
if params['mode'] == 'train':
    for i in range(args.run):
        agent.set_seed(params['env']['seed'] + i)
        agent.fit_model()
        if args.plot:
            max_abs_q_log.append(agent.max_abs_q_log)
            tderr_log.append(agent.tderr_log)
            total_avg_score_log.append(agent.total_avg_score_log)
            run_avg_score_log.append(agent.run_avg_score_log)
            if step_log is None or eps_log is None:
                step_log = agent.step_log
                eps_log = agent.eps_log
    if args.plot:
        # start plotting
        if 'log_lstm_grad' in params and params['log_lstm_grad']:
            plt.subplot(2, 2, 1)
            plt.plot(step_log, agent.grad_mean_ih)
            plt.title('Mean LSTM grad magnitude ih')
            plt.xlabel('Steps')
            plt.subplot(2, 2, 2)
            plt.plot(step_log, agent.grad_mean_hh)
            plt.title('Mean LSTM grad magnitude hh')
            plt.xlabel('Steps')
            plt.subplot(2, 2, 3)
            plt.plot(step_log, agent.grad_max_ih)
            plt.title('Max LSTM grad magnitude ih')
            plt.xlabel('Steps')
            plt.subplot(2, 2, 4)
            plt.plot(step_log, agent.grad_max_hh)
            plt.title('Max LSTM grad magnitude hh')
            plt.xlabel('Steps')
            plt.show()
        # max abs q
        plt.grid(True)
        max_abs_q_log = np.array(max_abs_q_log)
        max_abs_q_log = max_abs_q_log.max(axis=0).squeeze()
        plt.title('Max Absolute Q over steps')
        plt.ylabel('Max abs(Q)')
        plt.xlabel('Steps')
        plt.plot(step_log, max_abs_q_log)
        plt.show()
        plt.title('Histogram of log scale max abs(Q)')
        plt.ylabel('Counts')
        plt.xlabel('max abs(Q)')
        plt.hist(max_abs_q_log)
        plt.show()
        # tderr
        tderr_log = np.array(tderr_log)
        tderr_mean, tderr_var = tderr_log.mean(axis=0), tderr_log.var(axis=0)
        plt.plot(step_log, tderr_mean)
        plt.fill_between(step_log, tderr_mean - tderr_var, tderr_mean + tderr_var)
        plt.title('TD error over steps')
        plt.ylabel('TD error')
        plt.xlabel('Steps')
        plt.show()
        # total avg score
        min_len = min(len(total_avg_score_log[i]) for i in range(args.run))
        total_avg_score_log = [log[:min_len] for log in total_avg_score_log]
        eps_log = eps_log[:min_len]
        total_avg_score_log = np.array(total_avg_score_log)
        total_avg_score_log_mean, total_avg_score_log_var = total_avg_score_log.mean(axis=0), total_avg_score_log.var(axis=0)
        plt.plot(eps_log, total_avg_score_log_mean)
        plt.fill_between(eps_log, total_avg_score_log_mean - total_avg_score_log_var, total_avg_score_log_mean + total_avg_score_log_var)
        plt.title('Total average scores over episodes')
        plt.ylabel('Scores')
        plt.xlabel('Episodes')
        plt.show()
        # tderr
        run_avg_score_log = [log[:min_len] for log in run_avg_score_log]
        run_avg_score_log = np.array(run_avg_score_log)
        run_avg_score_log_mean, run_avg_score_log_var = run_avg_score_log.mean(axis=0), run_avg_score_log.var(axis=0)
        plt.plot(eps_log, run_avg_score_log_mean)
        plt.fill_between(eps_log, run_avg_score_log_mean - run_avg_score_log_var, run_avg_score_log_mean + run_avg_score_log_var)
        plt.title('Running average scores of window size %d over episodes' % agent.log_window_size)
        plt.ylabel('Scores')
        plt.xlabel('Episodes')
        plt.show()

elif params['mode'] == 'test':
    agent.test_model()
