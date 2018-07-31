# CS294-112 HW 1: Imitation Learning

Dependencies: PyTorch 0.4.0, MuJoCo version 1.31, OpenAI Gym 0.9.1


To generate training data by expert policy:  
python run_expert.py experts/Hopper-v1.pkl Hopper-v1 --num_rollouts 10 --save_dir data/

To run behavior cloning(supervised learning) :  
python behavior_cloning.py Hopper-v1 --hidden_dim 100 --weight_decay 1e-4 --batchsize 100 --epoch 50 --test_rollouts 5

To run DAgger :  
python dagger.py Hopper-v1 --hidden_dim 100 --weight_decay 1e-4 --batchsize 100 --dagger_iteration 10 --epoch 10 --test_rollouts 5

