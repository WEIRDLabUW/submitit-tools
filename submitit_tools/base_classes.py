from abc import ABC, abstractmethod
from dataclasses import asdict

import submitit
import os

import wandb

from submitit_configs import BaseRunConfig, WandbConfig
from typing import Union

from submitit_tools.create_objects import init_wandb


class JobBookKeeping:
    def __init__(self, run_config: BaseRunConfig, wandb_config: Union[WandbConfig, None]):
        self.run_config: BaseRunConfig = run_config
        self.wandb_config: Union[None, WandbConfig] = wandb_config
        self.job: Union[None, submitit.Job] = None
        self.result = None
        self.retries: int = 0

    def done(self):
        assert self.job is not None, "Called done on a book keeping that does not have a job"
        return self.job.done()

    def result(self):
        assert self.job is not None, "Called result on a book keeping that does not have a job"
        return self.job.result()


class BaseJob(ABC):
    """
    This class is what you should super class to create your own custom job.
    The methods you must overwrite are the __init__, __call__, and the checkpiont method.
    
    The __init__ method should take in a RunConfig and a WandbConfig 
    object and initalize your class with those values.
    
    The __call__ method is called once and contains the entire job
    
    the checkpoint method is called to save the state of the job. If the
    job is on the checkpoint partition, then this is not called, and only if timed out
    This means you MUST implement your own checkpointing.
    """

    @abstractmethod
    def __init__(self, run_config: BaseRunConfig, wandb_config: Union[WandbConfig, None]):
        os.makedirs(run_config.checkpoint_path, exist_ok=True)
        self.run_config: BaseRunConfig = run_config
        self.wandb_config: WandbConfig = wandb_config
        if os.path.exists(os.path.join(run_config.checkpoint_path, run_config.checkpoint_name)):
            self.loaded_checkpoint = False
        else:
            self.loaded_checkpoint = True

    @abstractmethod
    def __call__(self):
        if self.wandb_config is not None:
            init_wandb(self.wandb_config)
            wandb.config.update(asdict(self.run_config))

    def _save_checkpoint(self):
        """
        This is a helper method that you can use to save the state of your job. Make sure to call it
        periodically in your job incase it is killed and reaqueued.
        """
        pass

    def checkpoint(self, *args, **kwargs) -> submitit.helpers.DelayedSubmission:
        """
        This is a default method to checkpoint your job. This will run when the job times out
        and just resubmits the job with the same arguments that the job was created with. You should modify the
        _save_checkpoint() method instead of this one.
        """
        self._save_checkpoint()
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)
