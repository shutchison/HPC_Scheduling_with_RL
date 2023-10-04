import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces

from scheduler import Scheduler

import numpy as np

class HPCEnv(Env):
    def __init__(self):
        self.scheduler = Scheduler("machine_learning")
        self.action_space = spaces.MultiDiscrete([5, 6])
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,4))
        self.reward_range = (0, 1)
        self.spec = {}
        self.metadata = {"render_modes" : ["human"]}
        self.np_random = None

    def step(self, action):
        obs = .6
        reward = .8
        terminated = False
        truncated = False
        info = {}

        return (obs, reward, terminated, truncated, info)

    def reset(self, seed=42, options={}):
        obs = .6
        info = {}
        return (obs, info)

    def render(self):
        pass

    def close(self):
        pass

