import time
from dataclasses import dataclass

from submitit_tools.jobs import SubmititState, BaseJob, create_function_job, grid_search_job_configs
from submitit_tools.configs import SubmititExecutorConfig, BaseJobConfig, WandbConfig

#  This is the "TopLevel" Executor Config that submitit uses to execute all jobs.
@dataclass
class ExampleExecutorConfig(SubmititExecutorConfig):
    timeout_min: int = 4
    slurm_partition: str = "ckpt-all" # This is saying we want this to run on all possible nodes
    root_folder: str = "logging_dir"
    slurm_gpus_per_node: str = "1" # this is saying we want 1 gpu per node
    slurm_constraint:str = "l40s|a40|l40|a100" # This is saying we need a node with these gpus

# Since we do not need any checkpointing functionality, and the jobs will 
# not use the checkpoint path at all, we can use the base config.
job_configs = [BaseJobConfig for _ in range(10)]

# No wandb
wandb_configs = None

# This defines the function that our job should execute. This is used to create
# a function job. It takes in the job_cfg as a parameter (but doesn't use it in this case)
def job_fn(job_cfg: BaseJobConfig):
    from submitit_tools.jobs import run_nvidia_smi
    import re
    out = run_nvidia_smi()
    print(out)
    pattern =r'^\|\s+0\s+NVIDIA\s+(\S+)\s+'
    match = re.search(pattern, out, re.MULTILINE)
    return match.group(1).strip()

# This will return a job class that all it does is call and return the job_fn. We can
# pass this to the executor
GetNodeGPUJob = create_function_job(job_fn)

# 4. Create a Submitit state manager.
state = SubmititState(
    job_cls=GetNodeGPUJob,
    executor_config=ExampleExecutorConfig(),
    job_run_configs=job_configs,
    job_wandb_configs=wandb_configs,
)

# 5. Wait and then get the results of the submission
results = state.run_all_jobs()

# 6. Output the results. In this case, it is the gpu type
for result in results:
    print(result)
