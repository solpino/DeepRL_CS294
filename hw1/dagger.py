"""
Example usage:
    python dagger.py Hopper-v1 --render --hidden_dim 100 --weight_decay 1e-4 \
    --batchsize 100 --dagger_iteration 10 --epoch 10 --test_rollouts 5
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from torch.optim import Adam
from torch.autograd import Variable
from sklearn.utils import shuffle
import load_policy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--hidden_dim', type=int, default=100,
                        help='dim of hidden layer')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=100)
    parser.add_argument('--dagger_iterations', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--test_rollouts', type=int, default=5,
                        help='number of rollouts when test policy')
    
    
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    expert_policy = load_policy.load_policy("experts/"+args.task+".pkl")
    

    data = pickle.load(open("data/"+args.task+".p", "rb"))
    x = data['observations']
    y = data['actions']
    print ('dimension of obs :', x.shape)
    print('dimension of actions :', y.shape)
    
    x,y = shuffle(x,y)
    #split tarin/validation set
    num_train = int(len(x)*0.9)
    num_val = int(len(x)*0.1)
    x_train, y_train = x[:num_train],y[:num_train]
    x_val,y_val = x[num_train:],y[num_train:]
    
    class simple_model(nn.Module):
        def __init__(self,indim=100, hidden = 100,outdim = 100):
            super(simple_model, self).__init__()
            self.fc1 = nn.Sequential(
                                    nn.Linear(indim, hidden),
                                    nn.ReLU(),

                                    nn.Linear(hidden, hidden//2),
                                    nn.ReLU(),

                                    nn.Linear(hidden//2, hidden//4),
                                    nn.ReLU(),

                                    nn.Linear(hidden//4,outdim)
            )
            
        def obs_norm(self, obs_bo, obsnorm_mean, obsnorm_meansq):
            obsnorm_mean = torch.FloatTensor(obsnorm_mean)
            obsnorm_meansq = torch.FloatTensor(obsnorm_meansq)
            
            obsnorm_stdev = torch.sqrt(torch.max(torch.zeros(obsnorm_mean.size()), obsnorm_meansq - obsnorm_mean**2)).to(device)   
            normedobs_bo = (obs_bo - obsnorm_mean.to(device)) / (obsnorm_stdev + 1e-6)
            return normedobs_bo

        def forward(self, obs_bo,obsnorm_mean, obsnorm_meansq):
            return self.fc1(self.obs_norm(obs_bo,obsnorm_mean, obsnorm_meansq))
    
    
   
    
    model = simple_model(x.shape[1],  #input dim
                         args.hidden_dim, #hidden dim
                         y.shape[1]).to(device) #output dim
    
    
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),weight_decay = args.weight_decay)

    

    env = gym.make(args.task)
    #env.seed(1995)
    returns = []
    max_steps = env.spec.timestep_limit
    
    for i in range(args.dagger_iterations):
        print('iter', i)
        
        #retrain
        x_train,y_train = shuffle(x_train,y_train)
        for i in range(args.epoch):
            for idx in range(0,num_train,args.batchsize):
                x_batch, y_batch = torch.FloatTensor(x_train[idx:idx+args.batchsize]).to(device), torch.FloatTensor(y_train[idx:idx+args.batchsize]).to(device)
              
                y_ = model(x_batch,data['standardizer_mean'],data['standardizer_meansq'])
                loss = F.mse_loss(y_,y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        
        
        #generate observations
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        new_observations = []
        new_actions = np.empty((0, y.shape[1]))
        while not done:
            new_observations.append(obs)

            action = model(torch.FloatTensor(obs).to(device),data['standardizer_mean'],data['standardizer_meansq']).cpu().detach().numpy()
            obs, r, done, _ = env.step(action)

            totalr += r
            steps += 1
            #env.render()
            if steps >= max_steps:
                    break

        #generate actions of observations from policy_network
        print("generated train data :", len(new_observations))
        
        for idx in range(0,len(new_observations),args.batchsize):
            obs = torch.FloatTensor(new_observations[idx:idx+args.batchsize])
            actions = expert_policy(obs).detach().numpy()
            new_actions = np.append(new_actions, actions, axis=0)

        #aggregate
        x_train = np.append(new_observations,x_train,axis=0)
        y_train = np.append(new_actions,y_train,axis=0)


        print("total rewards: ", totalr)
        print("\n")
    
    
    
    #evaluate trained policy
    
    print ("evaluate trained policy")
    env = gym.make(args.task)
    returns = []
    observations = []
    actions = []
    max_steps = env.spec.timestep_limit
    for i in range(args.test_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            
            action = model(torch.FloatTensor(obs).to(device),data['standardizer_mean'],data['standardizer_meansq']).cpu().detach().numpy()
            obs, r, done, _ = env.step(action)

            totalr += r
            steps += 1
            if args.render:
                env.render()
            
            if steps >= max_steps:
                    break

        returns.append(totalr)
    
  

    print ('\n' + args.task)

    print('\n <dagger policy>')
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    print('\n <expert policy>')
    print('mean return', np.mean(data['returns']))
    print('std of return', np.std(data['returns']))
if __name__ == '__main__':
    main()