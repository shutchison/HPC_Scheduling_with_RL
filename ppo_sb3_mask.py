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
import argparse
from distutils.util import strtobool
import os 

import gymnasium as gym
import numpy as np

# Register the custom HPC environment with Gym
gym_id = "HPCEnv-v0"
gym.register(id=gym_id, entry_point="hpc_env:HPCEnv")


# Wrapping the constructor for MaskablePPO object to allow grid search of instantiation parameters.
# See docs here: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
# Default values for unprovided arguments are the same as the default from the 
# class definition
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--env', type=str, default="HPCEnv-v0",
        help='The environment to learn from (if registered in Gym, can be str)')
    parser.add_argument('--num-training-steps', type=int, default=100000,
        help='The number of steps you want to train for')
    parser.add_argument('--machines-csv', type=str, default='./data/machines.csv',
        help='The path to the machines csv file')
    parser.add_argument('--jobs-csv', type=str, default='./data/500_jobs.csv',
        help='The path to the jobs csv file')
    
    # Algorithm specific arguments
    parser.add_argument('--learning-rate', type=float, default=3e-4,
        help='The learning rate, it can be a function of the current progress remaining (from 1 to 0)')
    parser.add_argument('--n-steps', type=int, default=2048,
        help='The number of steps to run for each environment per update (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)')
    parser.add_argument('--batch-size', type=int, default=64,
        help='Minibatch size')
    parser.add_argument('--n-epochs', type=int, default=10,
        help="Number of epoch when optimizing the surrogate loss")   
    parser.add_argument('--gamma', type=float, default=0.99,
        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='Factor for trade-off of bias vs variance for Generalized Advantage Estimator')
    parser.add_argument('--clip-range', type=float, default=0.2,
        help="Clipping parameter, it can be a function of the current progress remaining (from 1 to 0).")
    parser.add_argument('--clip-range-vf', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='lipping parameter for the value function, it can be a function of the current progress remaining (from 1 to 0). This is a parameter specific to the OpenAI implementation. If None is passed (default), no clipping will be done on the value function. IMPORTANT: this clipping depends on the reward scaling.')
    parser.add_argument('--normalize-advantage', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggles advantages normalization")
    parser.add_argument('--ent-coef', type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
        help='the maximum norm for the gradient clipping')
    parser.add_argument('--target-kl', type=float, default=None,
        help='Limit the KL divergence between updates, because the clipping is not enough to prevent large update see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213) By default, there is no limit on the kl div.')   
    parser.add_argument('--verbose', type=int, default=0,  
        help='The verbosity level: 0 no output, 1 info, 2 debug')
    parser.add_argument('--seed', type=int, default=1,
        help='Seed of the experiment')
    args = parser.parse_args()
    return args


args = parse_args()
print(args)

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_mask()
env = gym.make(args.env, machines_csv=args.machines_csv, jobs_csv=args.jobs_csv)
env = ActionMasker(env, mask_fn) 

model = MaskablePPO(MaskableActorCriticPolicy, 
                    env, 
                    learning_rate=args.learning_rate,
                    n_steps=args.n_steps,
                    batch_size=args.batch_size,
                    n_epochs=args.n_epochs,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    clip_range=args.clip_range,
                    clip_range_vf=args.clip_range_vf,
                    normalize_advantage=args.normalize_advantage,
                    ent_coef=args.ent_coef,
                    vf_coef=args.vf_coef,
                    max_grad_norm=args.max_grad_norm,
                    target_kl=args.target_kl,
                    verbose=args.verbose,
                    seed=args.seed
                    )

experiment_name = f"{args.exp_name}_{args.jobs_csv.rstrip('.csv').lstrip('.')}_{args.num_training_steps}_{int(time.time())}"
tmp_path = f"./runs/{experiment_name}"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

model.set_logger(new_logger)

start_time = datetime.now()
model.learn(args.num_training_steps, progress_bar=True)

end_time = datetime.now()
print("learning complete.  Time spent: {}".format(end_time-start_time))

#thing = evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=-1000000, warn=False)
#print(thing)
#print("done evaluating")

jobs_csv_filename = os.path.splitext(os.path.split(args.jobs_csv)[1])[0]
model_save_path = f"./saved_models/{jobs_csv_filename}_{args.num_training_steps}"
model.save(model_save_path)
print(f"model saved to {model_save_path}.zip")
# del model # remove to demonstrate saving and loading
# model = MaskablePPO.load("ppo_mask")
# print("done saving and loading")

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
