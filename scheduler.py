import queue
from pprint import pprint
from machine import Machine
from job import Job
from datetime import datetime
import csv
import logging
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable
from sb3_contrib import MaskablePPO

QUEUE_DEPTH = 10 # Should agree with the queue depth the model was trained on

class Scheduler():
    def __init__(self, model_type:str) -> None: # what scheduling method to use
        self.machines = []
        self.global_clock = 0
        self.model_type = model_type

        #initialize self.future_jobs with all jobs we need to run
        self.future_jobs = queue.PriorityQueue()  # ordered based on submit time
        self.job_queue = []
        self.schedulable_jobs = []
        self.running_jobs = queue.PriorityQueue() # ordered based on end time
        self.completed_jobs = []
        self.failed_jobs = []

        # disabling logging to increase speed
        #logging.basicConfig(filename="output_files/simulation.log", level="DEBUG")
        #self.logger = logging.getLogger("Scheduling")

        self.csv_outfile_name = "output_files/test.csv"
        self.header_written = False

        self.num_total_jobs = 0
        self.temp_out_file = open(f"output_files/{self.model_type}.txt", "w")

        self.rl_model = None

    def conduct_simulation(self, machines_csv, jobs_csv, model_to_load="ppo_sb3_mask__1698091320"):
        self.load_machines(machines_csv)
        self.load_jobs(jobs_csv)
        print("Model is: {}".format(self.model_type))
        if self.model_type == "machine_learning":
            self.rl_model = MaskablePPO.load(model_to_load)
            print(f"Model loaded from {model_to_load}")
        
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
        # print(f"{len(self.machines)} machines loaded from {csv_file_name}")

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
        # print(f"{len(self.future_jobs.queue)} jobs loaded from {csv_file_name}")
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
                # self.logger.info("{} submitted at time {}".format(job.job_name, self.global_clock))
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
                            #self.logger.info("job {} ending at time {}".format(job.job_name, self.global_clock))
                            found = True
                            m.stop_job(job.job_name)
                            self.completed_jobs.append(job)
                            self.machines_log_status()
                            #self.log_training_data_csv(job, self.machines, m.node_name, "Stop")
                            break
                    if found:
                        break

        # Try to schedule any jobs which can now be run according to the proscribed algorithm
        self.schedule()

        # update global clock to be the next submisison or ending event
        if self.future_jobs.empty() and self.running_jobs.empty():
            print("No future jobs and no running jobs!")
            self.temp_out_file.close()
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


        # Prints for debugging algorithm
        self.temp_out_file.write("="*40)
        self.temp_out_file.write("\n")
        self.temp_out_file.write(f"Job queue depth is: {len(self.job_queue)}\n")
        for job in self.job_queue:
            self.temp_out_file.write(f"{job.job_name}: req_duration={job.req_duration:8} req_mem={job.req_mem:<12} req_cpus={job.req_cpus:<4} req_gpus={job.req_gpus}\n")
        for machine in self.machines:
            self.temp_out_file.write(f"{machine.node_name:<29}: avail_mem={machine.avail_mem:<10} avail_cpus={machine.avail_cpus:<2} avail_gpus={machine.avail_gpus}\n")
        
        if self.model_type == "sjf":
            self.job_queue.sort(key=lambda x: x.req_duration) # sjf = shortest job first by requested duration
            none_can_be_scheduled = False
            while not none_can_be_scheduled:
                any_scheduled = False
                for index, job in enumerate(self.job_queue):
                    assigned_machine = None
                    for m in self.machines:
                        if (m.avail_mem >= job.req_mem) and (m.avail_cpus >= job.req_cpus) and (m.avail_gpus >= job.req_gpus):
                            assigned_machine = m
                            break
                    if assigned_machine is not None:
                        self.temp_out_file.write(f"Starting job {job.job_name} on {assigned_machine.node_name}\n")
                        self.set_job_time(job)
                        assigned_machine.start_job(job)
                        self.temp_out_file.write(f"{assigned_machine.node_name:<29}: avail_mem={assigned_machine.avail_mem:<10} avail_cpus={assigned_machine.avail_cpus:<2} avail_gpus={assigned_machine.avail_gpus}\n")
                        #self.logger.info("job {} started at time {}".format(job.job_name, self.global_clock))
                        any_scheduled = True
                        self.running_jobs.put( (job.end_time, job) )
                        self.job_queue = self.job_queue[:index] + self.job_queue[index+1:]
                        #self.log_training_data_csv(job, self.machines, assigned_machine.node_name, "Start")
                        self.machines_log_status()
                        break
                if not any_scheduled:
                    none_can_be_scheduled = True

        elif self.model_type == "sjf_true":
            # Schedule the first job in the queue until the first job in the queue can not be 
            # scheduled, or the queue is empty.
            self.job_queue.sort(key=lambda x: x.req_duration) # sjf = shortest job first by requested duration

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
                    self.temp_out_file.write(f"Starting job {job.job_name} on {assigned_machine.node_name}\n")
                    self.set_job_time(job)
                    assigned_machine.start_job(job)
                    self.temp_out_file.write(f"{assigned_machine.node_name:<29}: avail_mem={assigned_machine.avail_mem:10} avail_cpus={assigned_machine.avail_cpus:<2} avail_gpus={assigned_machine.avail_gpus}\n")
                    
                    #self.logger.info("job {} started at time {}".format(job.job_name, self.global_clock))
                    any_scheduled = True
                    self.running_jobs.put( (job.end_time, job) )
                    self.job_queue = self.job_queue[1:]
                    #self.log_training_data_csv(job, self.machines, assigned_machine.node_name, "Start")
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
                        #self.logger.info("{} is unrunnable on any machine in the cluster".format(job.job_name, self.global_clock))
                        continue
                    break

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
                    self.temp_out_file.write(f"Starting job {job.job_name} on {assigned_machine.node_name}\n")
                    self.temp_out_file.write(f"{assigned_machine.node_name:<29}: avail_mem={assigned_machine.avail_mem:10} avail_cpus={assigned_machine.avail_cpus:<2} avail_gpus={assigned_machine.avail_gpus}\n")
                
                    self.set_job_time(job)
                    assigned_machine.start_job(job)

                    #self.logger.info("job {} started at time {}".format(job.job_name, self.global_clock))
                    any_scheduled = True
                    self.running_jobs.put( (job.end_time, job) )
                    self.job_queue = self.job_queue[1:]
                    #self.log_training_data_csv(job, self.machines, assigned_machine.node_name, "Start")
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
                        #self.logger.info("{} is unrunnable on any machine in the cluster".format(job.job_name, self.global_clock))
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
                    self.temp_out_file.write(f"Starting job {job.job_name} on {assigned_machine.node_name}\n")
                    self.temp_out_file.write(f"{assigned_machine.node_name:<29}: avail_mem={assigned_machine.avail_mem:10} avail_cpus={assigned_machine.avail_cpus:<2} avail_gpus={assigned_machine.avail_gpus}\n")
                
                    job = self.job_queue[best_job_index]
                    self.set_job_time(job)
                    assigned_machine.start_job(job)

                    #self.logger.info("job {} started at time {}".format(job.job_name, self.global_clock))
                    self.running_jobs.put( (job.end_time, job) )
                    self.job_queue = self.job_queue[:best_job_index] + self.job_queue[best_job_index+1:]
                    #self.log_training_data_csv(job, self.machines, assigned_machine.node_name, "Start")
                    self.machines_log_status()

                # No machine can run any job
                if min_fill_margin == 10:
                    return

        elif self.model_type == "machine_learning":
            if self.rl_model is None:
                print("You must load the model!")
                return
            
            while True:
                if len(self.job_queue) == 0:
                    self.temp_out_file.write("Queue Depth is 0, nothing to scheudule")
                    return
                
                self.update_scheduable_jobs()
                if len(self.schedulable_jobs) == 0:
                    self.temp_out_file.write("No scheudule jobs")
                    return

                for index, job_index_job_tuple in enumerate(self.schedulable_jobs[:10]):
                    job_index, job = job_index_job_tuple
                    self.temp_out_file.write(f"{job.job_name}: req_duration={job.req_duration:8} req_mem={job.req_mem:<12} req_cpus={job.req_cpus:<4} req_gpus={job.req_gpus}\n")
                
                for machine in self.machines:
                    self.temp_out_file.write(f"{machine.node_name:<29}: avail_mem={machine.avail_mem:<10} avail_cpus={machine.avail_cpus:<2} avail_gpus={machine.avail_gpus}\n")

                obs = self.get_obs(QUEUE_DEPTH, True)
                action_masks = self.get_action_mask(QUEUE_DEPTH)

                action, _states = self.rl_model.predict(obs, action_masks=action_masks)
                action = self.action_converter(action)
                
                job_index, machine_index = action
                # Confirm this is a valid job index and machine index
                if job_index > len(self.schedulable_jobs)-1 or machine_index > len(self.machines)-1:
                    print("="*40)
                    print(f"Action {action} appears to be invalid.")
                    print(f"{len(self.job_queue)} jobs in the queue.")
                    print(f"{len(self.machines)} machines in the cluster")
                    return 

                job_queue_index, job = self.schedulable_jobs[job_index]
                assigned_machine = self.machines[machine_index]

                # print(f"Trying to schedule {job}")
                # print(f"on machine {assigned_machine}")

                # Confirm this machine can actually run this job.  If not, do nothing
                if not assigned_machine.can_run(job):
                    print(f"{assigned_machine.node_name} lacks the resources to run {job.job_name}")
                    return

                # print(f"Starting {job} on {assigned_machine}")
                assigned_machine.start_job(job)
                self.set_job_time(job)
                #self.logger.info("job {} started at time {}".format(job.job_name, self.global_clock))
                self.running_jobs.put( (job.end_time, job) )
                self.job_queue = self.job_queue[:job_queue_index] + self.job_queue[job_queue_index+1:]
                #self.log_training_data_csv(job, self.machines, assigned_machine.node_name, "Start")
                self.machines_log_status()
                
            

        elif self.model_type == "oracle":
            self.job_queue.sort(key=lambda x: x.actual_duration) # oracle = shortest job first by actual duration

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
                        self.temp_out_file.write(f"Starting job {job.job_name} on {assigned_machine.node_name}\n")
                        self.temp_out_file.write(f"{assigned_machine.node_name:<29}: avail_mem={assigned_machine.avail_mem:10} avail_cpus={assigned_machine.avail_cpus:<2} avail_gpus={assigned_machine.avail_gpus}\n")
                
                        self.set_job_time(job)
                        assigned_machine.start_job(job)

                        #self.logger.info("job {} started at time {}".format(job.job_name, self.global_clock))
                        any_scheduled = True
                        self.running_jobs.put( (job.end_time, job) )
                        self.job_queue = self.job_queue[:index] + self.job_queue[index+1:]
                        #self.log_training_data_csv(job, self.machines, assigned_machine.node_name, "Start")
                        self.machines_log_status()
                        break
                if not any_scheduled:
                    none_can_be_scheduled = True

        elif self.model_type == "oracle_true":
            # Schedule the first job in the queue until the first job in the queue can not be 
            # scheduled, or the queue is empty.
            self.job_queue.sort(key=lambda x: x.actual_duration) # sjf = shortest job first by requested duration
            
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
                    self.temp_out_file.write(f"Starting job {job.job_name} on {assigned_machine.node_name}\n")
                    self.temp_out_file.write(f"{assigned_machine.node_name:<29}: avail_mem={assigned_machine.avail_mem:10} avail_cpus={assigned_machine.avail_cpus:<2} avail_gpus={assigned_machine.avail_gpus}\n")
                
                    self.set_job_time(job)
                    assigned_machine.start_job(job)

                    #self.logger.info("job {} started at time {}".format(job.job_name, self.global_clock))
                    any_scheduled = True
                    self.running_jobs.put( (job.end_time, job) )
                    self.job_queue = self.job_queue[1:]
                    #self.log_training_data_csv(job, self.machines, assigned_machine.node_name, "Start")
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
                        #self.logger.info("{} is unrunnable on any machine in the cluster".format(job.job_name, self.global_clock))
                        continue
                    break
                
        else:
            print(f"Invalid model_type: {self.model_type}")
            return

    def rl_schedule(self, action):
        # Advance time until a scheduling deciison can be made, or perform the 
        # action should be (job_queue_index, machine_index)
        # will assign the job at index job_queue_index to the machine at
        # machine_index
        # returns True if the simulation has more to do, False if there is nothing left to do.
        
        #if not isinstance(action, Iterable):
        action = self.action_converter(action)
        
        more_to_do = self.rl_tick()

        if not more_to_do:
            return False

        job_index, machine_index = action
        # Confirm this is a valid job index and machine index
        if job_index > len(self.schedulable_jobs)-1 or machine_index > len(self.machines)-1:
            print("="*40)
            print(f"Action {action} appears to be invalid.")
            print(f"{len(self.job_queue)} jobs in the queue.")
            print(f"{len(self.machines)} machines in the cluster")
            
            more_to_do = True
            return more_to_do

        job_queue_index, job = self.schedulable_jobs[job_index]
        assigned_machine = self.machines[machine_index]

        # print(f"Trying to schedule {job}")
        # print(f"on machine {assigned_machine}")

        # Confirm this machine can actually run this job.  If not, do nothing
        if not assigned_machine.can_run(job):
            print(f"{assigned_machine.node_name} lacks the resources to run {job.job_name}")
            more_to_do = True
            return more_to_do

        # print(f"Starting {job} on {assigned_machine}")
        assigned_machine.start_job(job)
        self.set_job_time(job)
        #self.logger.info("job {} started at time {}".format(job.job_name, self.global_clock))
        self.running_jobs.put( (job.end_time, job) )
        self.job_queue = self.job_queue[:job_queue_index] + self.job_queue[job_queue_index+1:]
        #self.log_training_data_csv(job, self.machines, assigned_machine.node_name, "Start")
        self.machines_log_status()
        self.update_scheduable_jobs()

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

    def rl_tick(self):
    # rl_tick will advance time until another scheudling action can occur, or
    # the simulation ends.
    # Returns True if there is more to do, False if the simulation has
    # completed.
        while True:
            any_job_can_run = False

            for job in self.job_queue:
                for machine in self.machines:
                    if machine.can_run(job):
                        any_job_can_run = True
                        break
                if any_job_can_run:
                    more_to_do = True
                    self.update_scheduable_jobs()
                    return more_to_do
            
            # move jobs who have been submitted now into the job_queue
            while not self.future_jobs.empty():
                first_submit = self.future_jobs.queue[0][0]
                if first_submit > self.global_clock:
                    break
                elif first_submit == self.global_clock:
                    job = self.future_jobs.get()[1]
                    #self.logger.info("{} submitted at time {}".format(job.job_name, self.global_clock))
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
                                #self.logger.info("job {} ending at time {}".format(job.job_name, self.global_clock))
                                found = True
                                m.stop_job(job.job_name)
                                self.completed_jobs.append(job)
                                self.machines_log_status()
                                #self.log_training_data_csv(job, self.machines, m.node_name, "Stop")
                                break
                        if found:
                            break

            any_schedulable_jobs = False
            for job in self.job_queue:
                for machine in self.machines:
                    if machine.can_run(job):
                        any_schedulable_jobs = True
                        break
                if any_schedulable_jobs:
                    # There is more to do at the current time step, so return
                    more_to_do = True
                    self.update_scheduable_jobs()
                    return more_to_do

            if self.future_jobs.empty() and self.running_jobs.empty() :
                # print("No future jobs, no running jobs!")
                self.update_scheduable_jobs()
                return False

            # update global clock to be the next submisison or ending event
            first_submit = 1e100
            first_end = 1e100
            if not self.future_jobs.empty():
                first_submit = self.future_jobs.queue[0][0]
            if not self.running_jobs.empty():
                first_end = self.running_jobs.queue[0][0]

            self.global_clock = min(first_submit, first_end)

            if self.global_clock == 1e100:
                print("Something has gone wrong updating the global clock.")
                self.update_scheduable_jobs()
                return False

            # Print status information to ensure progress happening
            #if len(self.completed_jobs) % 1000 == 0:
            #    print(f"{len(self.completed_jobs)}/{self.num_total_jobs} jobs completed")
            #    print(f"Current queue depth is {len(self.job_queue)}")

    def set_job_time(self, job):
        job.start_time = self.global_clock
        job.end_time = self.global_clock + job.actual_duration

    def update_scheduable_jobs(self):
        # Updates the list of tuples of jobs which are scheduable given the current 
        # resources available on the machines in cluster
        # [(job_queue_index, job), (job_queue_index, job]
        self.schedulable_jobs = []
        for job_index, job in enumerate(self.job_queue):
            for machine in self.machines:
                if machine.can_run(job):
                    self.schedulable_jobs.append( (job_index, job) )
                    break
        
    def get_obs(self, queue_depth_to_look, already_updated_schedulable=False):
        obs = []

        if not already_updated_schedulable:
            self.update_scheduable_jobs()

        # Only look so deep in the queue and pad if not enough jobs in the job queue
        for i in range(queue_depth_to_look):
            try:
                job_index, job = self.schedulable_jobs[i]
                obs.append(job.req_mem)
                obs.append(job.req_cpus)
                obs.append(job.req_gpus)
                obs.append(job.req_duration)
            except IndexError:
                obs.extend([0, 0, 0, 0])

            # if i < len(self.job_queue):
            #     job = self.job_queue[i]
            #     obs.append(job.req_mem)
            #     obs.append(job.req_cpus)
            #     obs.append(job.req_gpus)
            #     obs.append(job.req_duration)
            # else:
            #     obs.extend([0, 0, 0, 0])
        for machine in self.machines:
            obs.append(machine.avail_mem)
            obs.append(machine.avail_cpus)
            obs.append(machine.avail_gpus)
        
        return np.array(obs)

    def action_converter(self, action):
        # If action is a tuple or list, converts from 
        # (job_queue_index, machine_index) into the index of the 1d action space
        # If action is an int, converts from 1d action space index into
        # (job_queue_index, machine_index)

        # 1d action space is a list
        # [0, 0] = 0
        # [0, 1] = 1
        # [1, 0] = num_machines * 1 + 0
        num_machines = len(self.machines)

        #if not isinstance(action, Iterable):
        job_queue_index = action//num_machines
        machine_index = action - (job_queue_index * num_machines)
        return [job_queue_index, machine_index]
        # else:
            
        #     return num_machines * action[0] + action[1]

    def get_action_mask(self, queue_depth: int):
        # This allows us to mask off invalid actions (i.e. job/machine assignments which
        # are not valid).  Returns a list of booleans
        # action [0,1] would indicate assigning the job at index 0 to the machine at index 1
        # True if machines[1] can run job_queue[0], False if it cannot
        # This should be the same dimension as the action space
        # (default_queue_depth * 4) + (num_machines * 3)
        # 4 features per job up to the queue depth (req_mem, req_cpus, req_gpus, req_duration)
        # 3 features per machine (avail_mem, avail_cpus, avail_gpus)
        
        #print(f"specified queue depth is {queue_depth}")
        masks = []
        for i in range(queue_depth):
            job_valid = False
            try:
                job_index, job = self.schedulable_jobs[i]
                job_valid = True
            except IndexError:
                # There is no job at this positions in the job_queue,
                # so it's not a valid scheduling decision
                job = None
                pass
            for machine_index, machine in enumerate(self.machines):
                can_run = False
                if not job_valid:
                    can_run = False
                else:    
                    if machine.can_run(job):
                        can_run = True
                    else:
                        can_run = False
                #print(f"[{i}, {machine_index}] can run? {can_run}")
                masks.append(can_run)
        # for index, job in enumerate(self.job_queue):
        #     print(f"job_queue[{index}] is {job}")
        
        observation_space_length = (queue_depth * 4) + (len(self.machines) * 3)
        action_space_length = queue_depth * len(self.machines)
        if len(masks) != action_space_length:
            print("Error: Masks should have the same length as the action space")
            print(f"action_space_length={action_space_length}, len(masks)={len(masks)}")
        return masks

    def calculate_metrics(self) -> float:
        # returns a tuple (avg_queue_time, avg_clutser_util) 

        # iterate through self.completed_jobs and compute the avg queue time for all jobs which have been compelted
        queue_sum = sum([job.start_time-job.submission_time for job in self.completed_jobs])
        if len(self.completed_jobs) != 0:
            avg_queue_time = queue_sum/len(self.completed_jobs)
        else:
            avg_queue_time = 0

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

    def print_info(self):
        print("="*40)
        print(f"num future_job     = {len(self.future_jobs.queue)}")
        print(f"num job_queue      = {len(self.job_queue)}")
        print(f"num scheduable jobs= {len(self.schedulable_jobs)}")
        print(f"num running_jobs   = {len(self.running_jobs.queue)}")
        print(f"num completed_jobs = {len(self.completed_jobs)}")

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
