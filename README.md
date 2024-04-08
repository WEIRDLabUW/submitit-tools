# submitit-tools
This repository aims to give a simple way to submit jobs to Hyak integrating in weights and biases. You can install it as a library
but right now it is intended to be a submodule in your repository

## Installation
Run the following commands to install it as a submodule:
```bash
git submodule add https://github.com/WEIRDLabUW/submitit-tools.git
cd submitit-tools
pip install -e .
```



## Usage:

To see examples that are concrete, **run** `python scripts/example_torch_run.py`. This is a good entry point and will train up
9 mnest networks.

When you are using this as a submodule, you should only need to define the job class, run config and change config paramaters
when you instantiate them. 
### More details below


To use this for your own code, you first need to create a RunConfig. There are examples that 
you can look at in the `submitit_configs/run_config.py` file. Here is an example:

```python
from submitit_configs import BaseRunConfig
from dataclasses import dataclass

@dataclass
class ExampleRunConfig(BaseRunConfig):
    first_number: int = 1
    second_number: int = 2
    
    def __post_init__(self):
        self.checkpoint_path = f"add_{self.first_number}_to_{self.second_number}"
```
Then you have to create a job config which handles the core of your job:
```python
from submitit_tools import BaseJob
import os
import torch

class CustomJob(BaseJob):
    def __init__(self, run_config, wandb_config):
        # Initalizes some wandb stuff
        super().__init__(run_config, wandb_config)
        # Initalize all of the fields that you want to checkpoint
        
                
    def __call__(self):
        super().__call__()
        # Since no gpu in the init, have to load things here
        if not self.loaded_checkpoint:
            checkpoint = torch.load(os.path.join(self.run_config.checkpoint_path, self.run_config.checkpoint_name))
            # Do stuff with the checkpoint
            self.loaded_checkpoint = True
                    
        # Your job goes here, make sure to call the self._save_checkpoint() method
        # if wandb_config was not none, you can safely call wandb.log or other wandb functions 
        return result
    
    def _save_checkpoint(self):
        # So that we do not overide a real checkpoint with a random init model
        if not self.loaded_checkpoint:
            return
        # Save the checkpoing.
```
Then create an executor 
```python
from submitit_configs import SubmititExecutorConfig
from submitit_tools import create_executor
config = SubmititExecutorConfig(root_folder="mnest_submitit_logs",
                                    slurm_name="submitit-test",
                                    timeout_min=60*2,
                                    cpus_per_task=16,
                                    mem_gb=24)
executor =  create_executor(config)
```

Then you can create an executor state which handles the job submission and management:
```python
from submitit_tools import  SubmititState

state = SubmititState(
    executor # The executor configued you have created before
    job_cls # The job class you have defined
    job_run_configs # The list of run configs you want to run
    job_wandb_configs # The list of wandb configs you want to run
    with_progress_bar # If you want to use tqdm or not
    max_retries # The number of times you can resubmit the job if it times out or fails
    num_concurrent_jobs # The number of jobs you want to run concurrently
)

```

Then you can just use the runstate to monitor the progress and then process the results at the end. 
## Notes and todos:
-  Test if the jobs are recoverable if interrupted or stopped 
-  Handle job crashing vs slurm errors differently

### Notes:
If a job crashes, there is no way to distinguish this between being preempted or just crashed
which can probably lead to problems. 

We hypothisize that if a job is prempted, we have to handle the requing,
but if it is just timed out at 4 hours, submitit will manage it