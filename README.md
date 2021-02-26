# drqn-study

This repo implements the experimentation-oriented codebase for studying the effects of different algorithmic components on Deep Recurrent Q-Learning model. For the detail of the underlying theoretical base, please refer to the accompanying research paper.

## Installation

The codebase could be used for experiment in both Ubuntu and Window environments, where PyTorch is supported.

### Ubuntu

It is recommended to create a virtual environments before start installing the dependencies.
Please create a virtual environment at your home folder (or any other path):

```bash
cd ~
virtualenv drqn
source drqn/bin/activate
```

Then start install the dependencies in `requirements.txt`:

```bash
cd <path-to-project>/drqn_study
pip install -r requirements.txt
```

Finally, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install PyTorch according to your CUDA version associated with your GPU driver version.
The CUDA version could be checked with:

```bash
nvcc --version
```

### Windows

For Window, we have to install Anaconda first by following this [guide](https://docs.anaconda.com/anaconda/install/windows/).
Then please create an separated environment on Anaconda and install the dependencies:

```bash
cd <path-to-project>/drqn_study
pip install -r requirements.txt
```

For some reasons, the `atari-py` is not fully supported on Window. You may have to install a custom `atari-py` dependency here:
```
pip install -f https://github.com/Kojoley/atari-py/releases atari_py
```

Finally, please follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install PyTorch on Anaconda environment.

## Usage

This repo is designed with factory pattern that enables easy changing component settings for training study.
In general, the settings are stored in a `yaml` file in the folder `drqn_study\configs`.

The example `cartpole_dqn.yaml` config file is described with comments below:

```yaml
env:
  game: 'CartPole-v1' # the game environment name specified in Gym. For other games, just input the name defined in https://gym.openai.com/envs/#classic_control
  stack_len: 4 # the number of stack frame as an input to the training model
  seed: 1314121
  solved_criteria: 195 # the threshold defines the solving criteria for the environment.
  q_threhold: 100 # the threshold defines the solving episode reward criteria for the environment.
  pomdp: True # the flag turns the POMDP modification on or off
  pomdp_type: 'delete_dim'  # flickering | delete_dim
  pomdp_mask: [1, 0, 1, 0]  # this mask is chosen depended on the hiding dimensions and the observation shape of the env (i.e. 0)
  pomdp_prob: 0.8 # the probability used in flickering POMDP modification. 0.8 means 80% of the frames are presented (not zeroed).
  mode: 0

model:
  model_file: 'cartpole_dqn.pt' # the name to store the trained model.
  hidden_dim: 32 # hidden size of the model
  enable_dueling: False # this enables dueling mechanism in the network (untested)
  dueling_type: 'avg' # three types: avg | max | naive
  dtype: 'torch.float32' # data type of all numerical data while training

memory:
  size: 10000  # experience replay memory size
  a: 0  # the parameters for prioritized experience replay  (a=0, b=0 mean no prioritization)
  b: 0

env_type: 'gym' # gym | atari (please choose 'atari' if the game is atari type)
model_type: 'dqn_fc' # dqn_cnn | dqn_fc | drqn_cnn | drqn_fc (please choose 'drqn_' network if the agent is 'drqn')
memory_type: 'random' # episodic | random
agent_type: 'dqn' # dqn | drqn
value_criteria: 'mse_loss' # smooth_l1_loss | mse_loss (loss types)
optimizer: 'Adam' # Adam | Adagrad | RMSprop (optimizer types)
mode: 'train'  # train|test
train_visualize: False  # set True if you want to see the training visualization
retrain: True # whether to retrain completely from scratch or load model_file and continue training
solved_stop: True # whether to stop training after solved_criteria is reached

random_eps: 0 # initial random exploration episodes
episodes: 500 # total training episodes
steps: 200000 # total training steps
learn_start: 1000  # start learning after this learn_start steps
gamma: 0.99
# clip_grad: 1.0
lr: 0.001
lr_decay: False
eps_start: 1.0
eps_decay: 500
eps_end: 0.01
prog_freq: 50  # report training statistics after 50 episodes
target_model_update: 500 # if update target model after 500 steps
batch_size: 16
train_interval: 1 # triggering backward update after 1 steps
log_window_size: 100 # log window statistics
test_nepisodes: 1
log_step_interval: 100
log_episode_interval: 10
use_tensorboard: False
bootstrap_type: 'double_q'  # learn_q | target_q| double_q
```

You can place the `cartpole_dqn.yaml` into the `drqn_study\configs` folder and run the following command to start training:

```bash
python main.py cartpole_dqn.yaml --run 1 --plot True
```

where `--run` specifies the total of training times for averaging the statistics.
In general, after training complete, the model is saved in `drqn_study\saves`, the data and image statistics are saved in `drqn_study\logs`.
