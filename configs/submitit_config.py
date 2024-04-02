from dataclasses import dataclass


@dataclass
class SubmititExecutorBaseConfig:
    # Don't changet these
    slurm_gpus_per_node = 1  # The number of gpus that the job will take
    slurm_gpus_per_task = 1  # The number of gpus per task
    slurm_ntasks_per_node = 1  # The number of tasks per node


@dataclass
class SubmititExecutorConfig(SubmititExecutorBaseConfig):
    root_folder: str = "default_root_folder"  # This is the root folder to where submitit output is saved
    slurm_account = "weirdlab"  # This is the account to which the job is ran from
    slurm_name = "experiment"  # This is the name of the job that shows up on squeue
    timeout_min = (4 * 60) - 1  # This is the timeout in minutes
    cpus_per_task = 4  # This is the number of cpus per task
    mem_gb = 10  # This is the amount of ram required
    slurm_partition = "ckpt"  # This is the partition to which the job is submitted
    slurm_array_parallelism = 10
