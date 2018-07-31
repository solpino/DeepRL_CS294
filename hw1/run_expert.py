#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Hopper-v1.pkl Hopper-v1 --render \
            --num_rollouts 10 --save_dir data/

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import numpy as np
import gym
import load_policy
import torch

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--save_dir', type=str, default='data',
                        help='location to save training data')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')
    
    
    

    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = (policy_fn(torch.FloatTensor(obs))).detach().numpy() 
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('\n mean return :', np.mean(returns))
    print('std of return :', np.std(returns))
    print('obs shape :', np.array(observations).shape)
    print('action shape :', np.array(actions).shape)

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions),
                  'returns':  np.array(returns),
                  'standardizer_mean' : np.array(policy_fn.obsnorm_mean),
                  'standardizer_meansq' : np.array(policy_fn.obsnorm_meansq) 
                  }

    print('generated number of samples :', len(expert_data['observations']))

    #save train data
    pickle.dump(expert_data, open(args.save_dir+args.envname+".p", "wb"))
    
if __name__ == '__main__':
    main()
