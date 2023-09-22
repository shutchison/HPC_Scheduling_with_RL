from scheduler import Scheduler

s = Scheduler(model_type="sjf")
s.conduct_simulation("./data/more_machines.csv", "./data/all_jobs_202103.csv")
