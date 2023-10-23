import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces

from scheduler import Scheduler

import numpy as np
from enum import Enum

# MACHINES_CSV = "./data/machines.csv"
# JOBS_CSV = "./data/low_util.csv"
# JOBS_CSV = "./data/500_jobs.csv"

# MACHINES_CSV = "./data/tiny_machines.csv"
# JOBS_CSV = "./data/fcfs_test_jobs.csv"

DEFAULT_QUEUE_DEPTH = 10

# What percentage of jobs have a choice of machine on which they could be scheduled over time?

class Metrics(Enum):
    AVG_QUEUE_TIME = 0
    AVG_CLUSTER_UTILIZATION = 1

CURRENT_METRIC = Metrics.AVG_QUEUE_TIME
# CURRENT_METRIC = Metrics.AVG_CLUSTER_UTILIZATION

class HPCEnv(Env):
    def __init__(self, machines_csv = None, jobs_csv = None):
        self.step_counter = 0
        self.scheduler = Scheduler("machine_learning")
        self.MACHINES_CSV = machines_csv
        self.JOBS_CSV = jobs_csv
        if self.MACHINES_CSV is None or self.JOBS_CSV is None:
            print("You must provide the machines_csv and the jobs_csv arguments")

        self.scheduler.load_machines(machines_csv)
        self.scheduler.load_jobs(jobs_csv)
        self.NUM_MACHINES = len(self.scheduler.machines)

        

        # The number of jobs in the queue that are in the observation space
        # is either DEFAULT_QUEUE_DEPTH (defined above) or the number of 
        # submitted jobs, whichever is smaller.
        # Is this even necessary?
        self.QUEUE_DEPTH = min(DEFAULT_QUEUE_DEPTH, len(self.scheduler.future_jobs.queue))

        #self.action_space = spaces.MultiDiscrete([self.QUEUE_DEPTH, self.NUM_MACHINES])
        self.action_space = spaces.Discrete(self.QUEUE_DEPTH * self.NUM_MACHINES)
        
        # 4 attributes per job: req_mem, req_cpus, req_gpus, req_duration
        # 3 attributes per node: avail_mem, avail_cpus, avail_gpus
        low = [0] * self.QUEUE_DEPTH * 4
        low.extend([0] * self.NUM_MACHINES * 3)

        MOST_MEM_REQ = max([j[1].req_mem for j in self.scheduler.future_jobs.queue])
        MOST_CPUS_REQ = max([j[1].req_cpus for j in self.scheduler.future_jobs.queue])
        MOST_GPUS_REQ = max([j[1].req_gpus for j in self.scheduler.future_jobs.queue])
        MOST_DURATION_REQ = max([j[1].req_duration for j in self.scheduler.future_jobs.queue])

        high = [MOST_MEM_REQ, MOST_CPUS_REQ, MOST_GPUS_REQ, MOST_DURATION_REQ] * self.QUEUE_DEPTH
        for machine in self.scheduler.machines:
            high.append(machine.total_mem)
            high.append(machine.total_cpus)
            high.append(machine.total_gpus)

        # self.print_obs(low)
        # self.print_obs(high)

        self.observation_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.int32)
        self.reward_range = (0, 1)
        self.spec = {}
        self.metadata = {"render_modes" : ["human"]}
        self.np_random = None

    def seed(self, seed):
        np.random.seed(seed)
        
    def step(self, action):
        self.step_counter += 1
        more_to_do = self.scheduler.rl_schedule(action)
        terminated = not more_to_do
        obs = self.scheduler.get_obs(self.QUEUE_DEPTH)

        # self.print_obs(obs)

        # Try 0 reward for steps, and only using episodic reward?
        reward = 0
    
        avg_queue_time, avg_cluster_util = self.scheduler.calculate_metrics()
        if CURRENT_METRIC == Metrics.AVG_QUEUE_TIME:
            reward = 0 - avg_queue_time
        elif CURRENT_METRIC == Metrics.AVG_CLUSTER_UTILIZATION:
            reward = avg_cluster_util

        truncated = False
        info = {}

        # if self.step_counter % 400 == 0:
        #     self.scheduler.print_info()

        # ppo implementation expecting the following to be returned from step:
        # next_obs, reward, done, info
        return (obs, reward, terminated, truncated, info)

    def action_mask(self):
        masks = self.scheduler.get_action_mask(DEFAULT_QUEUE_DEPTH)
        return masks

    def reset(self, seed=None, options={}):
        self.scheduler = Scheduler("machine_learning")
        self.scheduler.load_machines(self.MACHINES_CSV)
        self.scheduler.load_jobs(self.JOBS_CSV)
        self.NUM_MACHINES = len(self.scheduler.machines)
        self.step_counter = 0

        # Advance time until there's something to do
        self.scheduler.rl_tick()
        obs = self.scheduler.get_obs(self.QUEUE_DEPTH)

        info = {}

        return (obs, info)

    def render(self):
        pass

    def close(self):
        pass

    def print_obs(self, obs):
        job_num = 0
        for i in range(self.QUEUE_DEPTH):
            index = i * 4
            print(f"job {job_num}    : {obs[index]:12} {obs[index+1]:2} {obs[index+2]:2} {obs[index+3]:2}")
            job_num += 1
        machine_num = 0
        for i in range(self.NUM_MACHINES):
            index = (4 * self.QUEUE_DEPTH) + (i * 3)
            print(f"machine {machine_num}: {obs[index]:12} {obs[index+1]:2} {obs[index+2]:2}")
            machine_num += 1
