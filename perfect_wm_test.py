import gym
import numpy as np

class WorldModel:  # :)
    def __init__(self, env):
        self.env = env

    def step(self, a):
        qpos0, qvel0 = (
            self.env.sim.data.qpos.flat.copy(),
            self.env.sim.data.qvel.flat.copy(),
        )
        s, r, d, _ = self.env.step(a)
        self.env.set_state(qpos0, qvel0)
        return s, r, d, _
    
env = gym.make("Hopper-v3")
actions = np.random.randn(10, 3)

wm = WorldModel(env)

for _ in range(10):
    s0 = env.reset()
    for a in actions:
        s1, r, d, _ = wm.step(a)
        # print(s1)
        # print(env.step(a)[0])
        s1_, r_, d_, _ = env.step(a)
        print(np.allclose(s1, s1_), np.allclose(r, r_), np.allclose(d, d_))