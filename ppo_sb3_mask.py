from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env

import time
from datetime import datetime

import gymnasium as gym
import numpy as np

gym_id = "HPCEnv-v0"
gym.register(id=gym_id, entry_point="hpc_env:HPCEnv")

tmp_path = f"./runs/{gym_id}__{int(time.time())}"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_mask()

env = gym.make(gym_id)
env = ActionMasker(env, mask_fn) 

steps_per_episode = 500
num_episodes = 100
desired_steps = steps_per_episode * num_episodes

model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
model.set_logger(new_logger)

start_time = datetime.now()
model.learn(desired_steps, progress_bar=True)
end_time = datetime.now()
print("learning complete.  Time spent: {}".format(end_time-start_time))


#thing = evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=-1000000, warn=False)
#print(thing)
#print("done evaluating")

model.save("ppo_mask")
del model # remove to demonstrate saving and loading
model = MaskablePPO.load("ppo_mask")
print("done saving and loading")

obs, _ = env.reset()
while True:
    # Retrieve current action mask
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break
print(env.scheduler.calculate_metrics())

print("done")
