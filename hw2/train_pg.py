import logz
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from torch.optim import Adam
from torch.autograd import Variable
from sklearn.utils import shuffle

#============================================================================================#
# Utilities
#============================================================================================#

def build_mlp(
        input_size, 
        output_size,
        n_layers=2, 
        size=64, 
        activation='tanh',
        ):
    
    #========================================================================================#
    #                           ----------SECTION 3----------
    # Network building
    #
    # Your code should make a feedforward neural network (also called a multilayer perceptron)
    # with 'n_layers' hidden layers of size 'size' units. 
    # 
    # The output layer should have size 'output_size' and activation 'output_activation'.
    #
    # Hint: use tf.layers.dense
    #========================================================================================#
    
    if activation == 'relu':
        non_linear = nn.ReLU()
    else:
        non_linear = nn.Tanh()
        
        
    layers = []
    layers.append(nn.Linear(input_size,size))
    layers.append(non_linear)
    for i in range(n_layers-1):
        layers.append(nn.Linear(size,size))
        layers.append(non_linear)
    
    layers.append(nn.Linear(size,output_size))
    
    mlp = nn.Sequential(*layers)  
    
    return mlp


def pathlength(path):
    return len(path["reward"])


#============================================================================================#
# Policy Gradient
#============================================================================================#


def train_PG(
             exp_name='',
             env_name='CartPole-v0',
             n_iter=100, 
             gamma=0.99, 
             min_timesteps_per_batch=1000, 
             max_path_length=None,
             learning_rate=2e-2,
             reward_to_go=True, 
             animate=False, 
             logdir=None,
             normalize_advantages=True,
             nn_baseline=False, 
             seed=0, 
             # network arguments
             n_layers=1,
             size=32,
             activation = 'Tanh',
             
             #baseline_network arguments
             bl_learning_rate=1e-3,
             bl_n_layers=1,
             bl_size = 32,
             bl_activation = 'Tanh',
             bl_n_iter=1
            
             ):
    start = time.time()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)
    env.seed(seed)
    
    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    #========================================================================================#
    # Notes on notation:
    # 
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in the function
    # 
    # Prefixes and suffixes:
    # ob - observation 
    # ac - action
    # _no - this tensor should have shape (batch size /n/, observation dim)
    # _na - this tensor should have shape (batch size /n/, action dim)
    # _n  - this tensor should have shape (batch size /n/)
    # 
    # Note: batch size /n/ is defined at runtime, and until then, the shape for that axis
    # is None
    #========================================================================================#

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    
    
    ''' Do not need in PyTorch
    #========================================================================================#
    #                           ----------SECTION 4----------
    # Placeholders
    # 
    # Need these for batch observations / actions / advantages in policy gradient loss function.
    #========================================================================================#

    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    if discrete:
        sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) 
    else:
        sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32) 

    # Define a placeholder for advantages
    sy_adv_n = TODO
    
    '''
    
    #========================================================================================#
    #                           ----------SECTION 4----------
    # Networks
    # 
    # Make symbolic operations for
    #   1. Policy network outputs which describe the policy distribution.
    #       a. For the discrete case, just logits for each action.
    #
    #       b. For the continuous case, the mean / log std of a Gaussian distribution over 
    #          actions.
    #
    #      Hint: use the 'build_mlp' function you defined in utilities.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ob_no'
    #
    #   2. Producing samples stochastically from the policy distribution.
    #       a. For the discrete case, an op that takes in logits and produces actions.
    #
    #          Should have shape [None]
    #
    #       b. For the continuous case, use the reparameterization trick:
    #          The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
    #
    #               mu + sigma * z,         z ~ N(0, I)
    #
    #          This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
    #
    #          Should have shape [None, ac_dim]
    #
    #      Note: these ops should be functions of the policy network output ops.
    #
    #   3. Computing the log probability of a set of actions that were actually taken, 
    #      according to the policy.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ac_na', and the 
    #      policy network output ops.
    #   
    #========================================================================================#
    
    
    def sampling(ob,sy_logstd):
        
        sy_logit = mlp(ob)
        
        if discrete:
            # YOUR_CODE_HERE
            sy_probs = F.softmax(sy_logit)
            sy_sampled_ac = torch.multinomial(sy_probs, 1)
        else:
            # YOUR_CODE_HERE
            sy_std = torch.exp(sy_logstd)
            z = torch.normal(torch.zeros(sy_logit.size())).to(device)
            sy_sampled_ac = sy_logit + z*sy_std
        return sy_sampled_ac
    
    
    
    
    '''Loss is defined in last section : "Performing the Policy Update" 
    #========================================================================================#
    #                           ----------SECTION 4----------
    # Loss Function and Training Operation
    #========================================================================================#

    loss = TODO # Loss function that we'll differentiate to get the policy gradient.
    update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss) 
    
    '''
   #========================================================================================#
    #                           ----------SECTION 5----------
    # Optional Baseline
    #========================================================================================#

    if nn_baseline:
        baseline_prediction = build_mlp(ob_dim,1,bl_n_layers,bl_size,bl_activation).to(device)
        bl_optimizer = Adam(baseline_prediction.parameters(),lr= bl_learning_rate)

    #========================================================================================#
    # Training Loop
    #========================================================================================#
    
    mlp = build_mlp(ob_dim,ac_dim,n_layers,size,activation).to(device)
    sy_logstd = nn.Parameter(torch.zeros(1, ac_dim).to(device))
    optimizer = Adam(list(mlp.parameters())+ [sy_logstd],lr =learning_rate)
    total_timesteps = 0
    
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)
                ac = sampling(torch.FloatTensor(ob).to(device),sy_logstd)
                ac = ac.cpu().detach().numpy()[0]
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            path = {"observation" : np.array(obs), 
                    "reward" : np.array(rewards), 
                    "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        returns = [path["reward"].sum() for path in paths]
        average_returns = (np.mean(returns))
        
        print("average_rewards : ", average_returns)
        print("\n")

        if average_returns > env.spec.reward_threshold:
            print("task solved")

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Computing Q-values
        #
        # Your code should construct numpy arrays for Q-values which will be used to compute
        # advantages (which will in turn be fed to the placeholder you defined above). 
        #
        # Recall that the expression for the policy gradient PG is
        #
        #       PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]
        #
        # where 
        #
        #       tau=(s_0, a_0, ...) is a trajectory,
        #       Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
        #       and b_t is a baseline which may depend on s_t. 
        #
        # You will write code for two cases, controlled by the flag 'reward_to_go':
        #
        #   Case 1: trajectory-based PG 
        #
        #       (reward_to_go = False)
        #
        #       Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over 
        #       entire trajectory (regardless of which time step the Q-value should be for). 
        #
        #       For this case, the policy gradient estimator is
        #
        #           E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
        #
        #       where
        #
        #           Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.
        #
        #       Thus, you should compute
        #
        #           Q_t = Ret(tau)
        #
        #   Case 2: reward-to-go PG 
        #
        #       (reward_to_go = True)
        #
        #       Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
        #       from time step t. Thus, you should compute
        #
        #           Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        #
        #
        # Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
        # like the 'ob_no' and 'ac_na' above. 
        #
        #====================================================================================#

        # YOUR_CODE_HERE
        q_n = []
        if reward_to_go:
            for path in paths:
                qs=[]
                q=0
                for reward in reversed(path["reward"]):
                    q =  reward + q*gamma
                    qs.append(q)

                q_n = q_n + qs[::-1]     
        else:
            for path in paths:
                discounted_reward = [path["reward"][i]*(gamma**i) for i in range(pathlength(path))]
                q_n = q_n + [np.sum(discounted_reward)]*pathlength(path)

        q_n = torch.FloatTensor(q_n).to(device)
        
        #====================================================================================#
        #                           ----------SECTION 5----------
        # Computing Baselines
        #====================================================================================#

        if nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            # 
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current or previous batch of Q-values. (Goes with Hint
            # #bl2 below.)
            
            b_n = baseline_prediction(Variable(torch.FloatTensor(ob_no)).to(device)).squeeze(1)
            b_n =  torch.mean(q_n) + ((b_n - torch.mean(b_n)) / torch.std(b_n)) * torch.std(q_n)
            adv_n = q_n - b_n
        else:
            adv_n = q_n.clone()

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Advantage Normalization
        #====================================================================================#

        if normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1. 
            
            adv_n = (adv_n - torch.mean(adv_n)) / torch.std(adv_n)

        #====================================================================================#
        #                           ----------SECTION 5----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if nn_baseline:
            # ----------SECTION 5----------
            # If a neural network baseline is used, set up the targets and the inputs for the 
            # baseline. 
            # 
            # Fit it to the current batch in order to use for the next iteration. Use the 
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the 
            # targets to have mean zero and std=1. (Goes with Hint #bl1 above.)
            
            normalize_q_n = (q_n - torch.mean(q_n))/torch.std(q_n)
            
            for i in range(bl_n_iter):
                b_n = baseline_prediction(Variable(torch.FloatTensor(ob_no)).to(device)).squeeze(1)
                bl_loss = F.mse_loss(b_n,normalize_q_n)
                bl_optimizer.zero_grad()
                bl_loss.backward()
                bl_optimizer.step()

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        #====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on 
        # the current batch of rollouts.
        # 
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below. 
        
        
        sy_logit = mlp(Variable(torch.FloatTensor(ob_no)).to(device))
        if discrete :
            sy_logprob_n = -F.cross_entropy(sy_logit,torch.LongTensor(ac_na).to(device),reduce=False)
        else:         
            sy_std = torch.exp(sy_logstd)
            sy_logprob_n = -0.5*torch.sum((((sy_logit - torch.FloatTensor(ac_na).to(device))/sy_std)**2),dim =1)# Hint: Use the log probability under a multivariate gaussian. 


        weighted_negative_likelihoods = sy_logprob_n * adv_n 
        loss = -torch.mean(weighted_negative_likelihoods)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        if itr == 0:
            logz.G.first_row = True
        logz.dump_tabular()
        

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--activation', '-act', type=str, default='tanh')
    
    parser.add_argument('--bl_n_layers', '-bl_l', type=int, default=1)
    parser.add_argument('--bl_size', '-bl_s', type=int, default=32)
    parser.add_argument('--bl_activation', '-bl_act', type=str, default='tanh')
    parser.add_argument('--bl_learning_rate', '-bl_lr', type=float, default=1e-3)
    parser.add_argument('--bl_n_iter', '-bl_n', type=int, default=1)
    
    
    
    args = parser.parse_args()

    
    
    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        train_PG(
            exp_name=args.exp_name,
            env_name=args.env_name,
            n_iter=args.n_iter,
            gamma=args.discount,
            min_timesteps_per_batch=args.batch_size,
            max_path_length=max_path_length,
            learning_rate=args.learning_rate,
            reward_to_go=args.reward_to_go,
            animate=args.render,
            logdir=os.path.join(logdir,'%d'%seed),
            normalize_advantages=not(args.dont_normalize_advantages),
            nn_baseline=args.nn_baseline, 
            seed=seed,
            n_layers=args.n_layers,
            size=args.size,
            activation=args.activation,

            bl_n_layers =args.bl_n_layers,
            bl_size =args.bl_size,
            bl_activation = args.bl_activation,
            bl_learning_rate = args.bl_learning_rate,
            bl_n_iter = args.bl_n_iter,
            )
        

        

if __name__ == "__main__":
    main()