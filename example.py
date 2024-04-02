import time


import submitit


class TrainingCallable():
    def __init__(self, run_config) -> None:
        self.__ = _ 
        
    def __call__(self, a, b):
        time.sleep(20)
        return a + b
    
    def checkpoint(self, a, b):
       return runhelpers.DelayedSubmission(self, a, b)


def main():
    print("creating executor")
    jobs = []
    executor = submitit.AutoExecutor(folder="log_test")
    # set timeout in min, and partition for running the job
    executor.update_parameters(slurm_partition="gpu-a40",
                            slurm_account="weirdlab",
                            slurm_name="experiment",
                            timeout_min=4,
                            mem_gb=10,
                            slurm_gpus_per_node=1,
                            slurm_gpus_per_task=1,
                            slurm_ntasks_per_node=1,
                            )
    executor.update_parameters(slurm_array_parallelism=10)
    with executor.batch():
        # In here submit jobs, and add them to the list, but they are all considered to be batched.
        for i in range(16):
            job = executor.submit(TrainingCallable(), 1, i)
            jobs.append(job)
        
    while jobs:
        time.sleep(1)
        done = []
        for i in range(len(jobs)):
            job = jobs[i]
            if job.done():
                done.append(i)
                print(job.result())
        for i in reversed(done):
            del jobs[i]
                
if __name__ == "__main__":
    main()
