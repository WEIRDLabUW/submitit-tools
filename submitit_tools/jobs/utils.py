import subprocess
import typing as tp
from itertools import product

import wandb.util
from submitit_tools.configs import BaseJobConfig, WandbConfig


from typing import TYPE_CHECKING


def run_nvidia_smi():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')

def update_to_unique_wandb_ids(wandb_configs: tp.List[tp.Union[WandbConfig, None]]):
    try:
        import wandb
    except:
        return
    for config in wandb_configs:
        if config:
            config.id = wandb.util.generate_id()


def grid_search_job_configs(config_dict:tp.Dict[str, tp.Union[tp.Any, tp.List]],
                            job_cls: tp.Union[tp.Type[BaseJobConfig], None],
                            job_creation_fn: tp.Union[tp.Callable, None] = None) -> tp.List[BaseJobConfig]:
    """This method grid searches over a list of parameter values and calls the create config method that is passed.
    The default one just creates the job config with the kwargs, and doesn't include a wandb config. You need
    to pass either a creation fn or a job cls

    Args:
        config_dict (tp.Dict[str, tp.Union[tp.Any, tp.List]]): This is a dictionary of arg name to a list of values or value
        job_cls (tp.Type[BaseJobConfig]): This is the job that takes those kwargs. It can be none if function provided
        job_creation_fn: This is a function that takes the exact kwargs of the configs, and returns what ever classes
         you want grid searched over.
        

    Returns:
        tp.List[BaseJobConfig]: Returns a list of the passed job config that are instantiated, or a grid search of
         results from the job_creation_fn
    """
    # If it is not a list make it a list:
    config_dict.update(
        (key, [value]) for key, value in config_dict.items() 
            if not isinstance(value, list)
    )

    result_configs = []
    
    # Generate all combinations of parameter values
    parameter_names = list(config_dict.keys())
    parameter_values = list(config_dict.values())
    combinations = list(product(*parameter_values))
    
    for combo in combinations:
        
        # Create a kwargs dictionary from the combo and names
        kwargs = dict(zip(parameter_names, combo))
        
        # Instantiate a job class with the kwargs
        if job_creation_fn is None: 
            job = job_cls(**kwargs)
        else:
            job = job_creation_fn(**kwargs)
        
        result_configs.append(job)
    return result_configs

