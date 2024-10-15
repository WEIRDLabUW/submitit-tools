import time
from dataclasses import dataclass

from submitit_tools.jobs import SubmititState, BaseJob
from submitit_tools.configs import SubmititExecutorConfig, BaseJobConfig, WandbConfig


# 1. This is the "TopLevel" JobConfig which will be the base
# for the programmatic modifications (e.g. to be swept).
@dataclass
class SimpleAddJobConfig(BaseJobConfig):
    first_number: int = 1
    second_number: int = 2

    # a way to interpolate a variable without using external libraries is to do it in __post_init__
    def __post_init__(self):
        self.checkpoint_path = f"add_{self.first_number}_to_{self.second_number}"


# 2. This is the "TopLevel" Executor Config that submitit uses to execute all jobs.
@dataclass
class ExampleExecutorConfig(SubmititExecutorConfig):
    timeout_min: int = 4
    slurm_partition: str = "ckpt"
    root_folder: str = "logging_dir"


# 3. Programmatically describe how you will vary each Base Config.
# For example, sweep the second number value from 0-9
# Instantiate a list of job configs with corresponding wandb configs.
job_configs = []
wandb_configs = []
for i in range(10):
    job_configs.append(
        SimpleAddJobConfig(
            first_number=0,
            second_number=i,
        ))
    wandb_configs.append(None)

# 3. Another way:
# Note, there is a built in utility that does this for you. You can define a custom 
# initialization method, or just use the default
from submitit_tools.jobs import grid_search_job_configs
params ={
    "first_number": 0,
    "second_number": [i for i in range(20)]
}

job_configs = grid_search_job_configs(params, job_cls=SimpleAddJobConfig)
wandb_configs = None    # Set the wandb configs to None instead of a list of None

# 4. Define your actual job
class SimpleAddJob(BaseJob):
    def __init__(self, job_config: SimpleAddJobConfig, wandb_config: None):
        super().__init__(job_config, wandb_config)
        # Initialize things here if you need to, but this does not run on the allocated node but instead your
        # main process

    def _initialize(self):
        # Since all of the information is in the configs, we do not need to
        # Initialize anything here.
        pass

    def _job_call(self):
        time.sleep(5)
        return self.job_config.first_number + self.job_config.second_number

    def _checkpoint(self):
        # Since this is a simple addition task, we do not need to checkpoint
        pass

# 4. Another way
# You can use the function job to do this for you (since this above job does not use the checkpointing functionality)
from submitit_tools.jobs import create_function_job
def job_fn(cfg: SimpleAddJobConfig):
    time.sleep(5)
    return cfg.job_config.first_number + cfg.job_config.second_number
job_cls = create_function_job(job_fn)


# 5. Create a Submitit state manager.
state = SubmititState(
    job_cls=SimpleAddJob,
    executor_config=ExampleExecutorConfig(),
    job_run_configs=job_configs,
    job_wandb_configs=wandb_configs,
    with_progress_bar=True,
    max_retries=1,
    num_concurrent_jobs=4
)

# 5. Run all of the jobs.
state.run_all_jobs()

# 6. Output the results.
for result in state.results:
    print(result)
