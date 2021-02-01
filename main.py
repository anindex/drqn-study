import yaml
import argparse
import logging
from datetime import datetime
from os import mkdir
from os.path import join, dirname, abspath

from src.utils.factory import EnvDict, ModelDict, MemoryDict, AgentDict
from src.utils.helpers import plot_lstm_grad_over_steps, plot_max_abs_q, plot_holistic_measure, save_data

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
loss_log = []  # per step
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
            loss_log.append(agent.loss_log)
            total_avg_score_log.append(agent.total_avg_score_log)
            run_avg_score_log.append(agent.run_avg_score_log)
            if step_log is None or eps_log is None:
                step_log = agent.step_log
                eps_log = agent.eps_log
    # prune out data
    min_eps_len = min(len(total_avg_score_log[i]) for i in range(args.run))
    total_avg_score_log = [log[:min_eps_len] for log in total_avg_score_log]
    run_avg_score_log = [log[:min_eps_len] for log in run_avg_score_log]
    eps_log = eps_log[:min_eps_len]
    min_step_len = min(len(max_abs_q_log[i]) for i in range(args.run))
    max_abs_q_log = [log[:min_step_len] for log in max_abs_q_log]
    loss_log = [log[:min_step_len] for log in loss_log]
    step_log = step_log[:min_step_len]
    data = {
        'step_log': step_log,
        'eps_log': eps_log,
        'max_abs_q_log': max_abs_q_log,
        'loss_log': loss_log,
        'total_avg_score_log': total_avg_score_log,
        'run_avg_score_log': run_avg_score_log
    }
    save_data(data, join(ROOT_DIR, 'logs', 'data', log_folder_name))
    if args.plot:
        # start plotting
        log_image_folder = join(ROOT_DIR, 'logs', 'images', log_folder_name)
        mkdir(log_image_folder)
        if 'log_lstm_grad' in params and params['log_lstm_grad']:
            plot_lstm_grad_over_steps(step_log, agent.grad_mean_ih, agent.grad_mean_hh, agent.grad_max_ih, agent.grad_max_hh, log_image_folder)
        plot_max_abs_q(step_log, max_abs_q_log, log_image_folder)
        plot_holistic_measure(step_log, loss_log, title='loss_over_steps', xlabel='Steps', ylabel='Loss', log_image_folder=log_image_folder)
        plot_holistic_measure(eps_log, total_avg_score_log, title='total_avg_score', xlabel='Episodes', ylabel='Scores', log_image_folder=log_image_folder)
        plot_holistic_measure(eps_log, run_avg_score_log, title='running_avg_score_windowsize%d' % agent.log_window_size, xlabel='Episodes', ylabel='Scores', log_image_folder=log_image_folder)
elif params['mode'] == 'test':
    agent.test_model()
