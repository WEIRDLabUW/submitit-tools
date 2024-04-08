from dataclasses import dataclass, field


@dataclass
class SubmititExecutorBaseConfig:
    # Don't changet these probably
    slurm_account: str = "weirdlab"  # This is the account to which the job is ran from
    slurm_ntasks_per_node = 1  # The number of tasks per node


@dataclass
class SubmititExecutorConfig(SubmititExecutorBaseConfig):
    root_folder: str = "default_root_folder"  # This is the root folder to where submitit output is saved
    slurm_name: str = "experiment"  # This is the name of the job that shows up on squeue
    timeout_min: int = (4 * 60) - 1  # This is the timeout in minutes
    cpus_per_task: int = 4  # This is the number of cpus per task
    mem_gb: int = 10  # This is the amount of ram required
    slurm_partition: str = "ckpt"  # This is the partition to which the job is submitted
    
    # This is the extra paramater dictionary where args become SBATCH commands
    slurm_additional_parameters: dict = field(default_factory= lambda: {
        "gpus": "a40:1" # This is gpu_type:number
    })
