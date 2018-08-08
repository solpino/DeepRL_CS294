# CS294-112 HW 2: Policy Gradient

Dependencies: PyTorch 0.4.0, MuJoCo version 1.31, OpenAI Gym 0.9.1  

Hyperparameter setting for InvertedPendulum-v1 without baseline netowrk :  
"activation"	:	"tanh",
"gamma"	:	0.99,
"learning_rate"	:	0.04,
"min_timesteps_per_batch"	:	1000,
"n_iter"	:	100,
"n_layers"	:	2,
"nn_baseline"	:	false,
"normalize_advantages"	:	true,
"reward_to_go"	:	true,
"seed"	:	2,
"size"	:	50

Hyperparameter setting for InvertedPendulum-v1 with baseline netowrk :  
"activation"	:	"tanh",
"bl_activation"	:	"tanh",
"bl_learning_rate"	:	0.005,
"bl_n_iter"	:	1,
"bl_n_layers"	:	2,
"bl_size"	:	50,
"gamma"	:	0.99,
"learning_rate"	:	0.01,
"n_iter"	:	100,
"n_layers"	:	2,
"nn_baseline"	:	true,
"normalize_advantages"	:	true,
"reward_to_go"	:	true,
"seed"	:	2,
"size"	:	50

Hyperparameter setting for HalfCheetah-v1 :  
"activation"	:	"relu",
"bl_activation"	:	"relu",
"bl_learning_rate"	:	0.006,
"bl_n_iter"	:	2,
"bl_n_layers"	:	2,
"bl_size"	:	50,
"gamma"	:	0.9,
"learning_rate"	:	0.04,
"max_path_length"	:	150.0,
"min_timesteps_per_batch"	:	5000,
"n_iter"	:	100,
"n_layers"	:	2,
"nn_baseline"	:	true,
"normalize_advantages"	:	true,
"reward_to_go"	:	true,
"seed"	:	20,
"size"	:	50
