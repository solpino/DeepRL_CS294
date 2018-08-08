import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import torch
import torch.nn as nn
import dqn
from dqn_utils import *
from atari_wrappers import *


class atari_model(nn.Module):
    def __init__(self,img_in, num_actions):
        super(atari_model, self).__init__()
       
        self.linear = nn.Sequential(
                        nn.Linear(img_in[0],256),
                        nn.ReLU(),
                        nn.Linear(256,128),
                        nn.ReLU(),
                        nn.Linear(128,64),
                        nn.ReLU(),
                        nn.Linear(64,num_actions)
        )
    


    def forward(self, x):
        return self.linear(x)
            

def atari_learn(env,
               num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0 
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=torch.optim.Adam,
        kwargs=1e-4,
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.2),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

  
    dqn.learn(
        env,
        func=atari_model,
        optimizer_spec=optimizer,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=1,
        target_update_freq=10000,
        grad_norm_clipping=10
)
    
    env.close()


def set_global_seeds(i):
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(i)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)


def get_env(seed):
    env = gym.make('Pong-ram-v0')

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind_ram(env)
   
    return env

def main():
    # Run training
    seed = 1 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(seed)
    atari_learn(env, num_timesteps=int(4e7))

if __name__ == "__main__":
    main()
