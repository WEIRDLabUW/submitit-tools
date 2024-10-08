from .base_classes import BaseJob
from submitit_tools.configs import BaseJobConfig
from typing import Callable, Any



class FunctionJob(BaseJob):
    """This class is a default job with the following behavior: 
        - It has a static field function_callback that needs to be set in the beginning
        - The function is a static class variable and all instances of the job will call it
        - It's only arguments can be your job config
    The purpose of this job is to make it so that if you want a job that does not checkpoint and on early termination
    it just requeues it without checkpointing, you can just use this and all you have to define is the job method
    """
    function_callback: Callable[[BaseJobConfig], Any] = None

    @classmethod
    def _set_function(cls, function_callback: Callable[[BaseJobConfig], Any]):
        cls.function_callback = staticmethod(function_callback)

    def _initialize(self):
        pass
    
    def _job_call(self):
        if self.function_callback is None:
            raise ValueError("Function not set. Use FunctionJob.set_function() before initializing jobs.")
        return self.function_callback(self.job_config)
    
    
def create_function_job(function_callback: Callable[[BaseJobConfig], Any]) -> FunctionJob:
    """This creates a specific function job class from a given function_callback.
    The created job will during the job call method just call the function with the
    job_config. 

    Args:
        function_callback (Callable[[BaseJobConfig], Any]): Your job. It can return anything
        but must take only the job_config as it's argument

    Returns:
        FunctionJob: A job class that is a function job but with your specific function set
    """ 
    class SpecificFunctionJob(FunctionJob):
        pass

    SpecificFunctionJob._set_function(function_callback=function_callback)
    return SpecificFunctionJob