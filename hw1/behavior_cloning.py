"""
Example usage:
    python behavior_cloning.py Hopper-v1 --render --hidden_dim 100 --weight_decay 1e-4 \
    --batchsize 100 --epoch 50 --test_rollouts 5
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--hidden_dim', type=int, default=100,
                        help='dim of hidden layer')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--test_rollouts', type=int, default=5,
                        help='number of rollouts when test policy')
    
    
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

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
    
    
   
    
    #training simple_model
    model = simple_model(x.shape[1],  #input dim
                         args.hidden_dim, #hidden dim
                         y.shape[1]).to(device) #output dim
    
    
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),weight_decay = args.weight_decay)


    for i in range(args.epoch):
        epoch_loss = 0
        for idx in range(0,num_train,args.batchsize):
            x_batch, y_batch = torch.FloatTensor(x_train[idx:idx+args.batchsize]).to(device), torch.FloatTensor(y_train[idx:idx+args.batchsize]).to(device)

            optimizer.zero_grad()
            y_ = model(x_batch,data['standardizer_mean'],data['standardizer_meansq'])

            loss = F.mse_loss(y_,y_batch)
            epoch_loss = epoch_loss + loss
            loss.backward()
            optimizer.step()
        print ("epoch :", i+1)
        print ("train loss: ", (epoch_loss/num_train).cpu().detach().numpy())

        epoch_loss = 0
        for idx in range(0,num_val,args.batchsize):
            x_batch, y_batch = x_val[idx:idx+args.batchsize],y_val[idx:idx+args.batchsize]
            x_batch, y_batch = torch.FloatTensor(x_batch), torch.FloatTensor(y_batch)
            x_batch, y_batch = x_batch.cuda(),y_batch.cuda()
            y_ = model(x_batch,data['standardizer_mean'],data['standardizer_meansq'])

            loss = F.mse_loss(y_,y_batch)
            epoch_loss = epoch_loss + loss
        print ("val loss: ", (epoch_loss/num_val).cpu().detach().numpy())
        print ("\n")



    #evaluate trained policy
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
            
            action = model(torch.FloatTensor(obs).to(device).cuda(),data['standardizer_mean'],data['standardizer_meansq']).cpu().detach().numpy()
            obs, r, done, _ = env.step(action)

            totalr += r
            steps += 1
            if args.render:
                env.render()
            
            if steps >= max_steps:
                    break

        returns.append(totalr)


    bc_mean=np.mean(returns)
    bc_std=np.std(returns)
    print ('\n' + args.task)
    print('\n <bc policy>')
    print('mean return', bc_mean)
    print('std of return', bc_std)

    print('\n <expert policy>')
    print('mean return', np.mean(data['returns']))
    print('std of return', np.std(data['returns']))
    

if __name__ == '__main__':
    main()