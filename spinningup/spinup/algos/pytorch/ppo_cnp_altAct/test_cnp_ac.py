import gym
import numpy as np
import torch

from core import CNPActorMLPCritic

# create environment
env = gym.make('Walker2d-v3')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
print("obs_dim: ", obs_dim)
print("act_dim: ", act_dim)

# create ac
ac = CNPActorMLPCritic(env.observation_space, 
                       env.action_space,
                          hidden_sizes=[64,32]
)

obs = env.reset()
context = np.random.rand(1, 11,  obs_dim+act_dim)


dists = ac.pi._distribution(
    context=torch.Tensor(context),
    obs=torch.Tensor(obs).reshape(1, 1, obs_dim)
)
print(dists)

ac.pi(
    observation=torch.Tensor(context),
    target=torch.Tensor(obs).reshape(1, 1, obs_dim)
)


# from core import mlp
# import time
# mymlp = mlp([12, 64,64,8], activation=torch.nn.Tanh)
# # loop
# start = time.time()
# for _ in range(4000):
#     obs = np.random.rand(1, 12)
#     pred = mymlp(torch.Tensor(obs))
#     true_val = np.random.rand(1, 8)
#     loss = torch.nn.functional.mse_loss(pred, torch.Tensor(true_val))
#     loss.backward()
# print("Time taken in loop: ", time.time() - start)
# start = time.time()
# obs = torch.zeros(4000, 12)
# for i in range(4000):
#     obs[i] = torch.Tensor(np.random.rand(1, 12))
# pred = mymlp(obs)
# true_val = torch.randn(4000, 8)
# loss = torch.nn.functional.mse_loss(pred, true_val)
# loss.backward()
# print("Time taken in parallel: ", time.time() - start)
