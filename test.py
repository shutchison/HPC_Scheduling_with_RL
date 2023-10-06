import gymnasium as gym
import hpc_env
from stable_baselines3 import PPO
from scheduler import Scheduler

NUM_STEPS = 75

gym.register(id="HPCEnv-v0", entry_point="hpc_env:HPCEnv")
env = gym.make("HPCEnv-v0")

env.reset()

for i in range(1, 10000):
    print("="*40)
    print(f"Step {i}")
    action = env.action_space.sample()

    print(f"action = {action}")
    print(env.action_space)
    obs, reward, terminated, truncated, info  = env.step(action)
    
    if i >= NUM_STEPS:
        truncated = True
    print(obs, reward, terminated, truncated, info)
    if terminated or truncated:
        break

# s = Scheduler("machine_learning")
# s.load_machines("./data/tiny_machines.csv")
# s.load_jobs("./data/fcfs_test_jobs.csv")
# s.rl_schedule([0, 1])
# print(s.machines[1].running_jobs)
# print(s)
