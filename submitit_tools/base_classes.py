from abc import ABC, abstractmethod
from configs.run_config import BaseRunConfig, WandbConfig
from typing import Union


        
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
    
    @abstractmethod
    def checkpoint(self):
        pass
    