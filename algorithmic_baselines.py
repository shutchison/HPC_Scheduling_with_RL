from scheduler import Scheduler
import time


for model in ["machine_learning", "fcfs", "sjf",  "bfbp", "oracle"]:
# queue_times = []
# model = "machine_learning"
# for i in range(10):
    # print("="*40)
    # print(f"Run {i}")
    print("="*40)
    s = Scheduler(model_type=model)
    s.conduct_simulation("./data/machines.csv", "./data/500_jobs.csv", model_to_load="./saved_models/500_jobs_300000")
    # s.conduct_simulation("./data/machines.csv", "./data/low_util.csv")
    avg_queue_time, avg_cluster_util = s.calculate_metrics()
    # queue_times.append(avg_queue_time)
    # s.print_info()
    print()
    
    #for job in s.failed_jobs:
    #    print(job)
    #time.sleep(5)
# avg_avg_queue_times = sum(queue_times)/len(queue_times)
# print(f"Avg. queue time avg was: {avg_avg_queue_times}")
