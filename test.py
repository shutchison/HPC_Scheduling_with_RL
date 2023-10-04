import gymnasium as gym
import hpc_env
from stable_baselines3 import PPO
from scheduler import Scheduler

#gym.register(id="HPCEnv-v0", entry_point="hpc_env:HPCEnv")
#env = gym.make("HPCEnv-v0")
#
#env.reset()
#
#for _ in range(10):
#    action = env.action_space.sample()
#    print(f"action = {action}")
#    obs, reward, terminated, truncated, info  = env.step(action)
#    print(obs, reward, terminated, truncated, info)

s = Scheduler("machine_learning")
s.load_machines("./data/tiny_machines.csv")
s.load_jobs("./data/fcfs_test_jobs.csv")
s.rl_schedule([0, 1])
print(s)
