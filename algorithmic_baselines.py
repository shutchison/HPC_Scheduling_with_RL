from scheduler import Scheduler
import time


for model in ["fcfs", "sjf", "bfbp", "oracle"]:
    print("="*40)
    s = Scheduler(model_type=model)
    s.conduct_simulation("./data/machines.csv", "./data/500_jobs.csv")
    #for job in s.failed_jobs:
    #    print(job)
    time.sleep(5)
    