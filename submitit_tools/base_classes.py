from abc import ABC, abstractmethod

import submitit

from configs.run_config import BaseRunConfig, WandbConfig
from typing import Union


class JobBookKeeping:
    def __init__(self, run_config: BaseRunConfig, wandb_config: Union[WandbConfig, None]):
        self.run_config: BaseRunConfig = run_config
        self.wandb_config: Union[None, WandbConfig] = wandb_config
        self.job: Union[None, submitit.Job] = None
        self.result = None
        self.retries: int = 0

    @property
    def done(self):
        assert self.job is not None, "Called done on a book keeping that does not have a job"
        return self.job.done()

    @property
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
        pass
    
    @abstractmethod
    def __call__(self):
        pass

    def checkpoint(self, *args, **kwargs) -> submitit.helpers.DelayedSubmission:
        """
        This is a default method to checkpoint your job. This will run when the job times out
        and just resubmits the job with the same arguments that the job was created with
        """
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)
    