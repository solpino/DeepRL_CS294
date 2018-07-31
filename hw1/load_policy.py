import pickle, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class load_policy(nn.Module):
    def __init__(self, filename):
        super(load_policy, self).__init__()
        self.filename = filename
        
        
        with open(filename, 'rb') as f:
            policy = pickle.loads(f.read())    
        
        self.obsnorm_mean = policy['GaussianPolicy']['obsnorm']['Standardizer']['mean_1_D']
        self.obsnorm_meansq = policy['GaussianPolicy']['obsnorm']['Standardizer']['meansq_1_D']    
        layer_params = policy['GaussianPolicy']['hidden']['FeedforwardNet']
        
        def read_layer(l):
            assert list(l.keys()) == ['AffineLayer']
            assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
            return l['AffineLayer']['W'].astype(np.float32), l['AffineLayer']['b'].astype(np.float32)
        
        layers = []
        for layer_name in sorted(layer_params.keys()):
                    l = layer_params[layer_name]
                    W, b = read_layer(l)
                    r, h = W.shape

                    layer = nn.Linear(r,h)

                    layer.weight.data.copy_(torch.from_numpy(W.transpose()))

                    layer.bias.data.copy_(torch.from_numpy(b.squeeze(0)))

                    layers.append(layer)

                    if 'lrelu' == policy['nonlin_type']:
                        layers.append(nn.LeakyReLU())
                    else:
                        layers.append(nn.Tanh())
        #output layer                
        W, b = read_layer(policy['GaussianPolicy']['out'])
        r,h =W.shape
        layer = nn.Linear(r,h)
        layer.weight.data.copy_(torch.from_numpy(W.transpose()))
        layer.bias.data.copy_(torch.from_numpy(b.squeeze(0)))
        layers.append(layer)

        self.policy_network = nn.Sequential(*layers)

    def obs_norm(self, obs_bo, obsnorm_mean, obsnorm_meansq):
        obsnorm_mean = torch.FloatTensor(obsnorm_mean)
        obsnorm_meansq = torch.FloatTensor(obsnorm_meansq)
        
        obsnorm_stdev = torch.sqrt(torch.max(torch.zeros(obsnorm_mean.size()), obsnorm_meansq - obsnorm_mean**2))        
        normedobs_bo = (obs_bo - obsnorm_mean) / (obsnorm_stdev + 1e-6)
        return normedobs_bo.squeeze(0)  
 
    def forward(self, obs):
        obs = self.obs_norm(obs,self.obsnorm_mean, self.obsnorm_meansq)
        ac = self.policy_network(obs)
        return ac