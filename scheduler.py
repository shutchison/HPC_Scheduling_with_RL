import queue
from pprint import pprint
from machine import Machine
from job import Job
from datetime import datetime
import csv
import logging
import matplotlib.pyplot as plt

class Scheduler():
    def __init__(self, model_type:str) -> None: # what scheduling method to use
        self.machines = []
        self.global_clock = 0
        self.model_type = model_type

        #initialize self.future_jobs with all jobs we need to run
        self.future_jobs = queue.PriorityQueue()  # ordered based on submit time
        self.job_queue = []
        self.running_jobs = queue.PriorityQueue() # ordered based on end time
        self.completed_jobs = []
        self.failed_jobs = []

        logging.basicConfig(filename="output_files/simulation.log", level="DEBUG")
        self.logger = logging.getLogger("Scheduling")

        self.csv_outfile_name = "output_files/test.csv"
        self.header_written = False

        self.num_total_jobs = 0

    def conduct_simulation(self, machines_csv, jobs_csv):
        self.load_machines(machines_csv)
        self.load_jobs(jobs_csv)
        print("Model is: {}".format(self.model_type))
        start_time = datetime.now()
        while True:
            if not self.tick():
                break
        end_time = datetime.now()
        print("Simulation complete.  Simulation time: {}".format(end_time-start_time))
        avg_queue_time, avg_cluster_util = self.calculate_metrics()
        print("Avg. Queue Time was {:,.2f} seconds".format(avg_queue_time))
        print("Avg. Clutser Util was {:,.2f}%".format(avg_cluster_util * 100))

        for m in self.machines:
            m.plot_usage(self.model_type)
        self.plot_clutser_usage()

    def machines_log_status(self):
        for m in self.machines:
            m.log_status(self.global_clock)

    def machines_plot(self):
        for m in self.machines:
            m.plot_usage(self.model_type)

    def reset(self, model_type:str):
        #chart_generation
        self.machines = []
        self.global_clock = 0
        self.model_type = model_type

        self.future_jobs = queue.PriorityQueue()  # ordered based on submit time
        self.job_queue = []
        self.running_jobs = queue.PriorityQueue() # ordered based on end time
        self.completed_jobs = []
        self.failed_jobs = []

        self.header_written = False
        self.num_total_jobs = 0

    def load_machines(self, csv_file_name):
        f = open(csv_file_name)
        lines = f.readlines()
        f.close()
        lines = list(map(str.strip, lines))
        # print(lines)
        headers = lines[0]
        for line in lines[1:]:
            elements = line.split(",")
            self.machines.append(Machine(elements[0], *map(int, elements[1:])))
        print(f"{len(self.machines)} machines loaded from {csv_file_name}")

    def load_jobs(self, csv_file_name):
        f = open(csv_file_name)
        lines = f.readlines()
        f.close()
        lines = list(map(str.strip, lines))
        # print(lines)
        headers = lines[0]
        for line in lines[1:]:
            elements = line.split(",")
            j = Job(elements[0], *map(int, elements[1:]))
            # wrap this in a tuple, so they are ordered by their sumbission time.
            self.future_jobs.put( (j.submission_time, j) )
        # initialize global clock to be the submission time of the first job
        self.global_clock = self.future_jobs.queue[0][0]
        print(f"{len(self.future_jobs.queue)} jobs loaded from {csv_file_name}")
        self.num_total_jobs = len(self.future_jobs.queue)

    def log_training_data_csv(self, job, machines, assignment, action):
        machine_data = []
        machine_headers = []

        # This is ugly, headers aren't being inserted yet
        ctr = 1
        for m in machines:
            machine_headers.append("Machine{}.node_name".format(ctr))
            machine_headers.append("Machine{}.total_mem".format(ctr))
            machine_headers.append("Machine{}.avail_mem".format(ctr))
            machine_headers.append("Machine{}.total_cpus".format(ctr))
            machine_headers.append("Machine{}.avail_cpus".format(ctr))
            machine_headers.append("Machine{}.total_gpus".format(ctr))
            machine_headers.append("Machine{}.avail_gpus".format(ctr))

            machine_data.append(m.node_name)
            machine_data.append(m.total_mem)
            machine_data.append(m.avail_mem)
            machine_data.append(m.total_cpus)
            machine_data.append(m.avail_cpus)
            machine_data.append(m.total_gpus)
            machine_data.append(m.avail_gpus)

            ctr = ctr+1

        job_data = []
        job_data.append(job.req_mem)
        job_data.append(job.req_cpus)
        job_data.append(job.req_gpus)
        job_data.append(job.req_duration)
        job_data.append(job.actual_duration)
        job_data.append(job.submission_time)
        job_data.append(job.start_time)
        job_data.append(job.end_time)

        headers = ["Clock", "Job", "Action", "Assignment", "Job.req_mem", "Job.req_cpus", "Job.req_gpus", "Job.req_duration", "Job.actual_duration", "Job.submission_time", "Job.start_time", "Job.end_time"] + machine_headers

        if not self.header_written:
            f = open(self.csv_outfile_name, "w")
            f.write(",".join(headers) + "\n")
            f.close()
            self.header_written = True

        with open(self.csv_outfile_name, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.global_clock, job.job_name, action] + [assignment] + job_data + machine_data)


    def tick(self):
        # iterate through self.future_jobs to find all jobs with job.submission_time == self.global_clock
        # move these jobs to self.job_queue ordered based on job.submision_time
        # iterate through self.running jobs and remove all jobs from machines whose job.end_time == self.global_clock
        # append these jobs to self.completed_jobs
        # iterate through self.job_queue and attempt to schedule all jobs using appropriate algorithm
        # move successfully scheduled jobs to the self.running_jobs

        # move jobs who have been submitted now into the job_queue
        while not self.future_jobs.empty():
            first_submit = self.future_jobs.queue[0][0]
            if first_submit > self.global_clock:
                break
            elif first_submit == self.global_clock:
                job = self.future_jobs.get()[1]
                self.logger.info("{} submitted at time {}".format(job.job_name, self.global_clock))
                self.job_queue.append(job)

        # stop all jobs who end at the current time and move them to completed
        while not self.running_jobs.empty():
            first_end = self.running_jobs.queue[0][0]
            if first_end > self.global_clock:
                break
            elif first_end == self.global_clock:
                end_time, job = self.running_jobs.get()
                found = False
                for m in self.machines:
                    for j in m.running_jobs:
                        if job.job_name == j.job_name:
                            self.logger.info("job {} ending at time {}".format(job.job_name, self.global_clock))
                            found = True
                            m.stop_job(job.job_name)
                            self.completed_jobs.append(job)
                            self.machines_log_status()
                            self.log_training_data_csv(job, self.machines, m.node_name, "Stop")
                            break
                    if found:
                        break

        # Try to schedule any jobs which can now be run according to the proscribed algorithm
        # This will do nothing if the model_type is "machine_learning"
        self.schedule()

        # update global clock to be the next submisison or ending event
        if self.future_jobs.empty() and self.running_jobs.empty():
            print("No future jobs and no running jobs!")
            return False

        first_submit = 1e100
        first_end = 1e100
        if not self.future_jobs.empty():
            first_submit = self.future_jobs.queue[0][0]
        if not self.running_jobs.empty():
            first_end = self.running_jobs.queue[0][0]

        self.global_clock = min(first_submit, first_end)

        if self.global_clock == 1e100:
            print("Something has gone wrong updating the global clock.")
            return False

        # Print status information to ensure progress happening
        #if len(self.completed_jobs) % 1000 == 0:
        #    print(f"{len(self.completed_jobs)}/{self.num_total_jobs} jobs completed")
        #    print(f"Current queue depth is {len(self.job_queue)}")

        return True

    def schedule(self):
        # schedule will use the specified scheduling algorithm (set by self.model_type)
        # and will schedule as many jobs as is currently possible using the proscribed algorithm
        # to schedule as many jobs as possible from the job queue and available machine resources.

        if self.model_type == "sjf":
            self.job_queue.sort(key=lambda x: x.req_duration) # sjf = shortest job first by requested duration

            none_can_be_scheduled = False
            while not none_can_be_scheduled:
                any_scheduled = False
                for index, job in enumerate(self.job_queue):
                    # scheduled, machine = self.shortest_job_first(job)

                    assigned_machine = None
                    for m in self.machines:
                        if (m.avail_mem >= job.req_mem) and (m.avail_cpus >= job.req_cpus) and (m.avail_gpus >= job.req_gpus):
                            assigned_machine = m
                            break
                    if assigned_machine is not None:
                        self.set_job_time(job)
                        assigned_machine.start_job(job)

                        self.logger.info("job {} started at time {}".format(job.job_name, self.global_clock))
                        any_scheduled = True
                        self.running_jobs.put( (job.end_time, job) )
                        self.job_queue = self.job_queue[:index] + self.job_queue[index+1:]
                        self.log_training_data_csv(job, self.machines, assigned_machine.node_name, "Start")
                        self.machines_log_status()
                        break
                if not any_scheduled:
                    none_can_be_scheduled = True

        elif self.model_type == "fcfs":
            # Schedule the first job in the queue until the first job in the queue can not be 
            # scheduled, or the queue is empty.
            while True:
                if len(self.job_queue) == 0:
                    return
                job = self.job_queue[0]
                assigned_machine = None
                for m in self.machines:
                    if (m.avail_mem >= job.req_mem) and (m.avail_cpus >= job.req_cpus) and (m.avail_gpus >= job.req_gpus):
                        assigned_machine = m
                        break

                if assigned_machine is not None:
                    self.set_job_time(job)
                    assigned_machine.start_job(job)

                    self.logger.info("job {} started at time {}".format(job.job_name, self.global_clock))
                    any_scheduled = True
                    self.running_jobs.put( (job.end_time, job) )
                    self.job_queue = self.job_queue[1:]
                    self.log_training_data_csv(job, self.machines, assigned_machine.node_name, "Start")
                    self.machines_log_status()
                else:
                    # Need a check in here if a job is un-runnable on any HPC machine, or we will fail to terminate
                    can_run_on_any_machine = False
                    for m in self.machines:
                        if (m.total_mem >= job.req_mem) and (m.total_cpus >= job.req_cpus) and (m.total_gpus >= job.req_gpus):
                            can_run_on_any_machine = True
                            break
                    if not can_run_on_any_machine:
                        self.job_queue = self.job_queue[1:]
                        self.failed_jobs.append(job)
                        self.logger.info("{} is unrunnable on any machine in the cluster".format(job.job_name, self.global_clock))
                        continue
                    break
        elif self.model_type == "bfbp":
            # Find the (job, machine) pairing which will result in the "fullest" machine, then start executing that job on that machine
            # Do this until there are no jobs left in the queue, or no more jobs will fit on any machine
            while True:
                if len(self.job_queue) == 0:
                    return

                min_fill_margin = 10
                assigned_machine = None
                best_job_index = None

                for job_index, job in enumerate(self.job_queue):
                    for m in self.machines:
                        # Check if this machine has enough reosources for this jobs.  If not, check the next machine
                        if (m.avail_mem < job.req_mem) or (m.avail_cpus < job.req_cpus) or (m.avail_gpus < job.req_gpus):
                            continue

                        # not all nodes have both GPUs and CPUs, so init each margin to 0
                        mem_margin = 0.0
                        cpu_margin = 0.0
                        gpu_margin = 0.0

                        # count how many attributes the node has to normalize the final margin
                        n_attributes = 0

                        if m.total_mem > 0:
                            mem_margin = (m.avail_mem - job.req_mem)/m.total_mem
                            n_attributes += 1

                        if m.total_cpus > 0:
                            cpu_margin = (m.avail_cpus - job.req_cpus)/m.total_cpus
                            n_attributes += 1

                        if m.total_gpus > 0:
                            gpu_margin = (m.avail_gpus - job.req_gpus)/m.total_gpus
                            n_attributes += 1

                        if n_attributes == 0:
                            print("{} has no virtual resources configured (all <= 0).".format(m.node_name))
                            fill_margin = 10
                        else:
                            fill_margin = (mem_margin + cpu_margin + gpu_margin)/n_attributes

                        # This (job, machine) combo results in a machine with the fewest resources left than we've found so far
                        if fill_margin < min_fill_margin:
                            min_fill_margin = fill_margin
                            assigned_machine = m
                            best_job_index = job_index

                # Start running the best job on the best machine
                if assigned_machine is not None and best_job_index is not None:
                    job = self.job_queue[best_job_index]
                    self.set_job_time(job)
                    assigned_machine.start_job(job)

                    self.logger.info("job {} started at time {}".format(job.job_name, self.global_clock))
                    self.running_jobs.put( (job.end_time, job) )
                    self.job_queue = self.job_queue[:best_job_index] + self.job_queue[best_job_index+1:]
                    self.log_training_data_csv(job, self.machines, assigned_machine.node_name, "Start")
                    self.machines_log_status()

                # No machine can run any job
                if min_fill_margin == 10:
                    return

        elif self.model_type == "machine_learning":
            return

        elif self.model_type == "oracle":
            return

        else:
            print(f"Invalid model_type: {self.model_type}")
            return

    def rl_schedule(self, action):
        # Advance time until a scheduling deciison can be made, or perform the 
        # action should be (job_queue_index, machine_index)
        # will assign the job at index job_queue_index to the machine at
        # machine_index
        more_to_do = self.rl_tick()
        if not more_to_do:
            return False

        job_index, machine_index = action
        # Confirm this is a valid job index and machine index
        if job_index > len(self.job_queue) or machine_index > len(self.machines):
            print(f"Action {action} appears to be invalid.")
            print(f"{len(self.job_queue)} jobs in the queue.")
            print(f"{len(self.machines)} machines in the cluster")
            more_to_do = True
            return more_to_do

        job = self.job_queue[job_index]
        assigned_machine = self.machines[machine_index]

        # Confirm this machine can actually run this job.  If not, do nothing
        if not machine.can_run(job):
            more_to_do = True
            return more_to_do

        assigned_machine.start_job(job)
        self.logger.info("job {} started at time {}".format(job.job_name, self.global_clock))
        self.running_jobs.put( (job.end_time, job) )
        self.job_queue = self.job_queue[:job_index] + self.job_queue[job_index+1:]
        self.log_training_data_csv(job, self.machines, assigned_machine.node_name, "Start")
        self.machines_log_status()

        # If there's a job that could run on any machine, we can make another
        # scheudling decision, so we can make another scheduling decision at
        # this time.  Return to let the RL agent decide what to do.
        any_job_can_run = False
        for job in self.job_queue:
            for machine in self.machines:
                if machine.can_run(job):
                    any_job_can_run = True
                    break
            if any_job_can_run:
                more_to_do = True
                return more_to_do

        # If there are no scheudling decisions that can be made, we need to
        # advance time to allow more jobs to be submitted, or allow some jobs
        # to finish.
        more_to_do = self.rl_tick()
        return more_to_do


    # rl_tick will advance time until another scheudling action can occur, or
    # the simulation ends.
    # Returns True if there is more to do, False if the simulation has
    # completed.
    def rl_tick(self):
        while True:
            any_job_can_run = False

            for job in self.job_queue:
                for machine in self.machines:
                    if machine.can_run(job):
                        any_job_can_run = True
                        break
                if any_job_can_run:
                    return True
            # self.tick will no schedule any jobs if the model_type is set to
            # "machine learning", but will advance time.
            more_to_do = self.tick()
            if not more_to_do:
                return False

    def set_job_time(self, job):
        job.start_time = self.global_clock
        job.end_time = self.global_clock + job.actual_duration

    def calculate_metrics(self) -> float:
        #returns a tuple (avg_queue_time, avg_clutser_util)

        # iterate through self.completed_jobs and compute the avg queue time for all jobs which have been compelted
        queue_sum = sum([job.start_time-job.submission_time for job in self.completed_jobs])
        if len(self.completed_jobs) != 0:
            avg_queue_time = queue_sum/len(self.completed_jobs)
        else:
            raise ValueError("There are no completed jobs!")

        utils = []
        for m in self.machines:
            utils.append(m.get_util())
        avg_cluster_util = sum(utils)/len(utils)

        return (avg_queue_time, avg_cluster_util)

    def plot_clutser_usage(self):
        cluster_mem  = 0
        cluster_cpus = 0
        cluster_gpus = 0

        for m in self.machines:
            cluster_mem  += m.total_mem
            cluster_cpus += m.total_cpus
            cluster_gpus += m.total_gpus

        cluster_avail_mem  = []
        cluster_avail_cpus = []
        cluster_avail_gpus = []

        for i in range(len(self.machines[0].avail_cpus_at_times)):
            current_time_avail_mem = 0
            current_time_avail_cpus = 0
            current_time_avail_gpus = 0

            for m in self.machines:
                current_time_avail_mem  += m.avail_mem_at_times[i]
                current_time_avail_cpus += m.avail_cpus_at_times[i]
                current_time_avail_gpus += m.avail_gpus_at_times[i]

            cluster_avail_mem.append(current_time_avail_mem)
            cluster_avail_cpus.append(current_time_avail_cpus)
            cluster_avail_gpus.append(current_time_avail_gpus)

        tick_times = self.machines[0].tick_times

        fig = plt.figure(figsize=[12,10])
        fig.suptitle(f"Cluster Utilization ({self.model_type})")
        mem_perc = [1 - mem/cluster_mem for mem in cluster_avail_mem]
        cpu_perc = [1- cpu/cluster_cpus for cpu in cluster_avail_cpus]
        if cluster_gpus != 0:
            gpu_perc = [1 - gpu/cluster_gpus for gpu in cluster_avail_gpus]
        else:
            gpu_perc = [0 for _ in cluster_avail_gpus]
        ticks = [datetime.fromtimestamp(t) for t in tick_times]

        plt.plot(ticks,
                 mem_perc,
                 color="red",
                 label="Memory Utilization")

        plt.plot(ticks,
                 gpu_perc,
                 color="blue",
                 label="GPU Utilization")

        plt.plot(ticks,
                cpu_perc,
                color="green",
                label="CPU Utilization")
        plt.legend()

        plt.xlabel("time")
        plt.ylabel("Percentage utilized")
        plt.savefig("plots/{}_Cluster.jpg".format(self.model_type), bbox_inches="tight")
        plt.close(fig)

    def __repr__(self):
        s = "Scheduler("
        for key, value in self.__dict__.items():
            if type(value) == queue.PriorityQueue:
                s += str(value.queue)
            else:
                s += str(key) + "=" + repr(value) + ",\n"
        return s[:-2] + ")"

    def __str__(self):
        s = "Scheduler("
        for key, value in self.__dict__.items():
            if type(value) == queue.PriorityQueue:
                s += str(key) + "=" + repr(value.queue) + ",\n"
            else:
                s += str(key) + "=" + repr(value) + ",\n"
        return s[:-2] + ")"
