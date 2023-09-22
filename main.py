from scheduler import Scheduler

s = Scheduler(model_type="sjf")
s.conduct_simulation("./data/machines.csv", "./data/high_util.csv")


