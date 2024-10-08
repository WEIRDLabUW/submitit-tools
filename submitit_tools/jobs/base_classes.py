from abc import ABC, abstractmethod
from dataclasses import asdict

import submitit
import os
import traceback
import wandb

from submitit_tools.configs import BaseJobConfig, WandbConfig
from typing import Union


class FailedJobState:
    "Class to represent a failed job"

    def __init__(self, e: Exception, stack_trace: str, job_config: BaseJobConfig,
                 wandb_config: Union[WandbConfig, None]):
        self.exception = e
        self.stack_trace = stack_trace
        self.job_config = job_config
        self.wandb_config = wandb_config

    def __str__(self):
        return f"Exception: {str(self.exception)}\n\nStack Trace:\n{self.stack_trace}"


class JobBookKeeping:
    def __init__(self, job_config: BaseJobConfig, wandb_config: Union[WandbConfig, None]):
        self.job_config: BaseJobConfig = job_config
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
    The methods you must overwrite are the __init__, __call__, and the checkpoint method.

    The __init__ method should take in a JobConfig and a WandbConfig. You can add extra logic
    here but it is not needed.

    Since the job is created and then pickled to the correct compute node, you must override the
    _initialize method to initialize your job with all of the train information

    The __call__ method is called once and contains the entire job

    the checkpoint method is called to save the state of the job. If the
    job is on the checkpoint partition, then this is not called, and only if timed out
    This means you MUST implement your own checkpointing.
    """

    def __init__(self, job_config: BaseJobConfig, wandb_config: Union[WandbConfig, None]):
        os.makedirs(job_config.checkpoint_path, exist_ok=True)
        self.job_config: BaseJobConfig = job_config
        self.wandb_config: WandbConfig = wandb_config
        self.initialized = False

    def __call__(self):
        """
        Handles  checkpointing and wandb logic, and then calls the user defined _call method
        """
        if self.wandb_config is not None:
            wandb.init(**asdict(self.wandb_config))
            wandb.config.update(asdict(self.job_config))

        if not self.initialized:
            try:
                self._initialize()
            except Exception as e:
                # Print the exception for logging purposes (moved to run_state)
                # print(e)
                # traceback.print_exc()
                # This means that the job failed and it was the user's fault, not submitit
                return FailedJobState(e, traceback.format_exc(), self.job_config, self.wandb_config)
            self.initialized = True

        try:
            return self._job_call()
        except Exception as e:
            # Print the exception for logging purposes (moved to the run_state)
            # print(e)
            # traceback.print_exc()
            # This means that the job failed and it was the user's fault, not submitit
            return FailedJobState(e, traceback.format_exc(), self.job_config, self.wandb_config)

    @abstractmethod
    def _job_call(self):
        """
        This method is called by the call method and contains the entire job
        """
        raise NotImplementedError("This method must be implemented")

    @abstractmethod
    def _initialize(self):
        """
        This method is called by the call method and initializes the object. Make sure to implement this
        with all the field definitions that you need. Once this method has been called, it is assumed safe to call
        ``_save_checkpoint`` at any time
        """
        raise NotImplementedError("This method must be implemented")

    def _save_checkpoint(self):
        """
        This is a helper method that you can use to save the state of your job. Make sure to call it
        periodically in your job incase it is killed and requeued. ONLY CALL THIS IN YOUR _job_call METHOD
        """
        pass

    def checkpoint(self, *args, **kwargs) -> submitit.helpers.DelayedSubmission:
        """
        This is a default method to checkpoint your job. This will run when the job times out
        and just resubmits the job with the same arguments that the job was created with. You should modify the
        _save_checkpoint() method instead of this one.
        """
        if self.initialized:
            self._save_checkpoint()
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)

    def checkpoint_exists(self):
        return os.path.exists(os.path.join(self.job_config.checkpoint_path, self.job_config.checkpoint_name))

    @property
    def save_path(self):
        """Helper method that returns the save path of the job. This is useful for saving / loading the job"""
        return os.path.join(self.job_config.checkpoint_path, self.job_config.checkpoint_name)