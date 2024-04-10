from spinup.algos.pytorch.ppo_altAct.ppo_altAct import PPOBuffer as Buffer1
from spinup.algos.pytorch.ppo_1proc.ppo_1proc import PPOBuffer as Buffer2
import numpy as np

# list of random actions
actions = np.random.randn(100, 3)
# list of random observations
obs = np.random.randn(100, 5)
# list of random rewards
rews = np.random.randn(100)
# list of random values
vals = np.random.randn(100)
# list of random logprobs
logprobs = np.random.randn(100)

# init buffers:
buffer1 = Buffer1(100,gamma=0.99,lam=0.95) # mine
buffer2 = Buffer2(
    obs_dim=5,
    act_dim=3,
    size=100,
    gamma=0.99,
    lam=0.95
) # original

# fill buffers:
for i in range(50):
    buffer1.store(obs[i], actions[i], rews[i], vals[i], logprobs[i],
    [],[],[],[])
    buffer2.store(obs[i], actions[i], rews[i], vals[i], logprobs[i])
buffer1.finish_path()
buffer2.finish_path()

for i in range(50,100):
    buffer1.store(obs[i], actions[i], rews[i], vals[i], logprobs[i],
    [],[],[],[])
    buffer2.store(obs[i], actions[i], rews[i], vals[i], logprobs[i])
buffer1.finish_path()
buffer2.finish_path()

# compare buffers:
print('Comparing buffers...')
res1 = buffer1.get()
res2 = buffer2.get()

print((res1["obs"]-res2["obs"]).max())
print((res1["act"]-res2["act"]).max())
print((res1["ret"]-res2["ret"]).max())
print((res1["adv"]-res2["adv"]).max())
print((res1["logp"]-res2["logp"]).max())

print(res1["adv"][:50])
print(res2["adv"][:50])