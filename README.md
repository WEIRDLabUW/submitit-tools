# submitit-tools

This repository aims to give a simple way to submit jobs to Hyak integrating in weights and biases. I personally find it very useful at a small scale, but it's main purpose is to manage and run lots and lots of hyak runs. You can install it as a library
or as a submodule in your repository. Any changes or bug fixes are welcome!

It is also useful to not take up all of shared lab resources and utilize only a specific number of gpus at a time.
## Installation
Run the following command to install this as a package.
```bash
pip install git+https://github.com/WEIRDLabUW/submitit-tools.git
```
Alternatively, run the following commands to install it as a submodule:
```bash
git submodule add git@github.com:WEIRDLabUW/submitit-tools.git
cd submitit-tools
pip install -e .
```

## Usage:

#### To see examples that are concrete, run `python scripts/example_torch_run.py`. This is a good entry point and will train up 9 mnest networks. Note you will need torch and torchvision in your conda environment.
There is also a good example with comments in `scripts/example_executor.py`.

For usage, you should only need to define a job class, run config. The rest is just instantiating configs and passing
them to submitit tools.

### Important things to note:
- You need to make sure that the checkpoint path is **unique for each run**. If it is not,
    it will just load the checkpointing from the previous run and then imediatly finish.
- You can use the results from the submitit state, and add no checkpointing functionality which 
    would work on runs on the weird-lab partitions where you know they won't be interrupted or prempted
- You do not have to use the WANDB config and can instead handle wandb yourself. If you choose to do so, make sure that you handle the checkpointing and resuming of WANDB runs. You can look into this codebase to get insight into how to do this


## Paramaters you want to change in the submitit executor config:
This is the base executor config with all of the paramaters. You don't usually have to touch all of them
```python
class SubmititExecutorConfig:
    root_folder: str = "default_root_folder"  # This is the root folder to where submitit logs are saved
    slurm_account: str = "weirdlab"  # This is the account to which the job is ran from i.e weirdlab or rselab
    slurm_ntasks_per_node: int = 1  # The number of tasks per node. You should keep this as 1 unless running distributed training
    slurm_gpus_per_node: str = "a40:1"  # The number of gpus that the node will have
    slurm_nodes: int = 1  # The number of nodes utilized
    slurm_name: str = "experiment"  # This is the name of the job that shows up on squeue
    timeout_min: int = (4 * 60) - 1  # This is the timeout of your job in minutes
    cpus_per_task: int = 4  # This is the number of cpus per task
    mem_gb: int = 10  # This is the amount of ram per node
    slurm_partition: str = "ckpt-all"  # This is the partition to which the job is submitted

    # This is the extra paramater dictionary where args become SBATCH commands. Probably do not need to use but anything you put
    # here will be written like --SBATCH key=value in the submitted job.
    slurm_additional_parameters: dict = field(default_factory=lambda: {})
    
    # These paramaters control the mail to aspects of slurm. They default to None which does not send any emails.
    slurm_mail_user: Union[str, None] = None  # overide with your email, e.g "jacob33@uw.edu"
    slurm_mail_type: Union[str, None] = None  # overide with the type of email you want, e.g "BEGIN,END"
```

#### Slurm GPU Paramaters:
- For the node types, if you want to request a node with a specific gpu number and type, you can do it with a list like this `type1:num,type2:num` where it will asign your job to the first availible node with gpus of that number and type. For example, to use all gpus on the ckpt-all that are powerful, I set `slurm_gpus_per_node="a40:1,l40:1,l40s:1,a100:1"`.

## Clear Code Examples:


To use this for your own code, you first need to create a RunConfig. All a run config represents is
the paramaters that you want to be able to pass your runs. They can be what ever you want

```python
from submitit_tools.configs import BaseJobConfig
from dataclasses import dataclass, field

@dataclass
class CustomJobConfig(BaseJobConfig):
    parameter1: int = 3
    parameter2: list = field(default_factory=lambda: [1, 2, 3])

    def __post_init__(self):
        # To check if you have not already specified the checkpoint. You can also exclude this
        # and make sure to override the checkpoint name.
        if self.checkpoint_name is "checkpoint.pt":
            self.checkpoint_name = f"checkpoint_{self.parameter1}.pt"
        self.checkpoint_path = f"your_beautiful_checkpoint_path"
```

The next step is to define the job that you want to run. This is what is duplicated on each
slurm job, and ran with the config you defined above. 

```python
from submitit_tools.jobs import BaseJob

class CustomJob(BaseJob):
    def __init__(self, run_config: CustomJobConfig, wandb_config):
        # Important, you must call the super init method
        super().__init__(run_config, wandb_config)
        
        # Note: This method runs synchronously on the node queuing all the jobs, so
        # Try not to do much at all in this method. You do not have to define this method
        
        # I like to type specify the config here to help pycharm's intelisense
        self.job_config: CustomJobConfig = run_config 

    def _initialize(self):
        # Write code to initialize all the fields in your class. This runs on the alocated node.
        self.custom_field = self.job_config.parameter2 * self.job_config.parameter1
        
        if self.checkpoint_exists():
            # Load the checkpoint and update your fields
            pass
       
                
    def _job_call(self):
        # Your job goes here, make sure to call the self._save_checkpoint() method
        # if wandb_config was not none, you can safely call wandb.log or other wandb functions 
        # Make sure to include the step in wandb.log() otherwise you might experience weird data stuff
        result = "Cool Stuff"
        return result
    
    def _save_checkpoint(self):
        # Save the checkpoint. Only call this method during your _job_call method
        pass
```

Then you can create an executor state which handles the job submission and management:

```python
from submitit_tools.jobs import SubmititState
from submitit_tools.configs import SubmititExecutorConfig
import time
class ExampleExecutorConfig(SubmititExecutorConfig):
    root_folder: str = "logging_dir"
    timeout_min: int = 4
    slurm_partition: str = "ckpt-all"
    slurm_gpus_per_node: str = "l40s:1,a40:1,l40:1"

config = ExampleExecutorConfig()
# Create your list of job configs and wandb configs. 
# They need to be the same length
job_configs, wandb_configs = generate_train_configs()
state = SubmititState(
    job_cls=CustomJob,
    executor_config=config,
    job_run_configs=job_configs,
    job_wandb_configs=wandb_configs,
    with_progress_bar=True,
    max_retries=4,
    num_concurrent_jobs=-1,
    cancel_on_exit=True
)

# Monitor the progress of your jobs. You can do more 
# complicated things here
while state.done() is False:
    state.update_state()
    time.sleep(1)

# Process the results. The results are updated as jobs complete 
# so you can access this before
for result in state.results:
    print(result)
```

## Notes and todos:
- The checkpoint partition is a little weird. You can find the documentation [here](https://hyak.uw.edu/docs/compute/checkpoint/)
- To contribute please branch and then submit a PR.
- Want to add a git automation and make this a pypy package
- ~~Handle job crashing vs slurm errors differently~~
- I think that it will crash a job if the checkpoint gets corrupted while being written and this will be unrecoverable
- ~~Figure out requesting multiple types of gpus for large checkpoint runs~~
- ~~Add functionality to cancel jobs if the executor dies, or the user wants to. Right now if the main file crashes, the jobs will still keep runing, just without  being requeued if needed.~~


