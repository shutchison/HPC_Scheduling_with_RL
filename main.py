from scheduler import Scheduler

s = Scheduler(model_type="sjf")
s.conduct_simulation("./data/machines.csv", "./data/low_util.csv")


