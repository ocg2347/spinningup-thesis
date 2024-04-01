import gym
import numpy as np

env = gym.make("Walker2d-v3")

act = env.action_space.sample()

s0 = env.reset()
qpos0, qvel0 = env.sim.data.qpos.flat.copy(), env.sim.data.qvel.flat.copy()
s0_computed = np.concatenate(
    [qpos0[1:], np.clip(qvel0, -10, 10)]
).ravel()  # this is correct logic
print("s0: ", s0)
print("qpos0: ", qpos0)
print("qvel0: ", qvel0)

print("-------------------------------------")

s1 = env.step(act)[0]
print("s1: ", s1)
print("qpos1: ", env.sim.data.qpos)
print("qvel1: ", env.sim.data.qvel)

print("-------------------------------------")

# set state to s0
env.set_state(qpos0, qvel0)
qpos0_, qvel0_ = env.sim.data.qpos.flat.copy(), env.sim.data.qvel.flat.copy()
print("qpos0_: ", qpos0_)
print("qvel0_: ", qvel0_)
s0_ = np.concatenate([qpos0_[1:], np.clip(qvel0_, -10, 10)]).ravel()

print("s0_: ", s0_)

print("-------------------------------------")
s1_ = env.step(act)[0]
print("s1_: ", s1_)
