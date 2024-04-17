import time
from dataclasses import dataclass

from submitit_tools import SubmititState, BaseJob
from submitit_configs import SubmititExecutorConfig, BaseJobConfig, WandbConfig


# 1. This is the "TopLevel" JobConfig which will be the base
# for the progrmatic modifications (e.g. to be sweeped). 
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
    partition: str = "ckpt"

# 3. Programatically describe how you will vary each Base Config.
# For example, sweep the second number value from 0-9
# Instantiate a list of job configs with corresponding wandb configs.
run_configs = []
wandb_configs = []
for i in range(10):
    run_configs.append(
        SimpleAddJobConfig(
            first_number = 0,
            second_number = i,
         ))
    wandb_configs.append(None)

# 3. This is the Job description.
# First, configure both:
#    RunConfig: what is passed to the Job
#    WandBConfig: how it is logged   
# Second, describe what happens on "__call__"
class SimpleAddJob(BaseJob):
    def __init__(self, job_config: SimpleAddJobConfig, wandb_config: WandbConfig):
        self.job_config = job_config
        self.wandb_config = wandb_config

    def __call__(self):
        time.sleep(5)
        return self.job_config.first_number + self.job_config.second_number
    
    def _initialize(self):
        pass

# 4. Create a Submitit state manager.
state = SubmititState(
    job_cls=SimpleAddJob,
    executor_config=ExampleExecutorConfig(), 
    job_run_configs=run_configs,
    job_wandb_configs=wandb_configs,
    with_progress_bar=True,
    max_retries=1,
    num_concurent_jobs=4
)

# 5. Keep track of state while all jobs are running.
while state.done() is False:
    state.update_state()
    time.sleep(1)

# 6. Output the results.
for result in state.results:
    print(result)
