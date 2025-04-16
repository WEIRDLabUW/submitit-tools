from dataclasses import asdict
import time
from typing import List, Type, Union, Any
import atexit

import submitit
from tqdm.auto import tqdm

from submitit_tools.configs import BaseJobConfig, WandbConfig, SubmititExecutorConfig
from submitit_tools.jobs.base_classes import BaseJob, JobBookKeeping, FailedJobState
from .utils import update_to_unique_wandb_ids

class SubmititState:
    """
    This class is used to store the state of the running jobs
    """

    def __init__(self,
                 job_cls: Type[BaseJob],
                 executor_config: SubmititExecutorConfig,
                 job_run_configs: List[BaseJobConfig],
                 job_wandb_configs: Union[List[Union[WandbConfig, None]], None],
                 with_progress_bar: bool = True,
                 output_error_messages: bool = True,
                 max_retries: int = 5,
                 num_concurrent_jobs: int = -1,
                 cancel_on_exit: bool = True
                 ):
        """
        Initializes the SubmititState instance.

        :param job_cls: The class of the job to be submitted. This should be a subclass of BaseJob.
        :param executor_config: Configuration for the executor. This should be an instance of ``SubmititExecutorConfig``
        :param job_run_configs: A list of configurations for each job to be run.
            Each configuration should be a derivation of  ``BaseJobConfig``.
        :param job_wandb_configs: A list of configurations for Weights & Biases for each job.
            Each configuration should be either an instance of ``WandbConfig`` or None.
            You can instead pass None if not using wandb
        :param with_progress_bar: A boolean indicating whether to display a progress bar for the jobs
        :param output_error_messages: A boolean indicating whether to output more verbose error messages
        :param max_retries: The maximum number of times to retry a job if it is interrupted
        :param num_concurrent_jobs: The number of jobs to run concurrently
        :param cancel_on_exit: A boolean indicating whether to cancel all jobs when the main process exits
        """
        if job_wandb_configs is None:
            job_wandb_configs = [None for _ in range(len(job_run_configs))]
        
        # update the unique IDs so that the runs can be resumed
        update_to_unique_wandb_ids(job_wandb_configs)
        assert len(job_run_configs) == len(job_wandb_configs), "The number of job configs and wandb configs must match"

        self._executor: submitit.AutoExecutor = self._init_executor_(executor_config)
        self._job_cls: Type[BaseJob] = job_cls
        self.max_retries: int = max_retries
        self.output_error_messages = output_error_messages
        self.progress_bar = tqdm(total=len(job_run_configs), desc="Job Progress") if with_progress_bar else None

        self._pending_jobs: List[JobBookKeeping] = [
            JobBookKeeping(job_config, wandb_config) for job_config, wandb_config in
            zip(job_run_configs, job_wandb_configs)
        ]

        self._running_jobs: List[Union[JobBookKeeping, None]] = [None] * (num_concurrent_jobs if num_concurrent_jobs > 0
                                                                         else len(self._pending_jobs))

        self._finished_jobs: List[JobBookKeeping] = []

        if cancel_on_exit:
            atexit.register(self._cancel_all)

        self._update_submitted_queue()


    def _init_executor_(self, config: SubmititExecutorConfig):
        """
        Private helper to initialize executor. We want to abstract this responsibility from user.
        """

        kwargs = asdict(config)
        debug_mode = kwargs.pop("debug_mode")

        if debug_mode:
            # debug so run locally
            executor = submitit.AutoExecutor(folder=kwargs.pop("root_folder"), cluster="debug")
            return executor
        
        # real run
        executor = submitit.AutoExecutor(folder=kwargs.pop("root_folder"))
        executor.update_parameters(**kwargs)

        return executor

    def _update_submitted_queue(self):
        """
        Private helper method to move jobs from pending to running if possible
        """
        for i in range(len(self._running_jobs)):
            if len(self._pending_jobs) == 0:
                # No pending jobs to add
                return

            if self._running_jobs[i] is not None:
                # We already have a running job or no pending left at this index so don't do anything
                continue
            # We have to queue a job if we can
            job = self._executor.submit(
                self._job_cls(self._pending_jobs[0].job_config, self._pending_jobs[0].wandb_config)
            )
            self._running_jobs[i] = self._pending_jobs.pop(0)
            self._running_jobs[i].job = job
            assert job is not None, "Job is None"

    def run_all_jobs(self, sleep_time: int =1) -> List[Any]:
        """This code will block and run/update this state until all the jobs are finished.
        This replaces the user having to make a while loop, and call the update_state() themselves.

        Args:
            sleep_time (int): How long to wait between calling update_state(). Defaults to 1. 
            When done, state.done() will be true
        Returns:
            List[Any]: The results from the job, same as the self.results attribute
        """
        while not self.done():
            self.update_state()
            time.sleep(sleep_time)
        return self.results
    
    def update_state(self):
        """
        This method is called to update the current state of this object.
        It both finishes completed jobs, as well as starts up the next jobs
        in the queue if possible
        """
        for i in range(len(self._running_jobs)):
            if self._running_jobs[i] is None:
                continue

            current_job = self._running_jobs[i]
            assert current_job.job is not None, "Job is running but job is None"

            if not current_job.done():
                continue

            # The job is finished! It might have failed:
            try:
                if current_job.job.num_tasks == 1:
                    result = [current_job.job.result()]
                else:
                    result = current_job.job.results()
                for r in result:
                    if isinstance(r, FailedJobState):
                        # If we get here, the job failed, and it is the user's fault (so don't requeue)
                        print(
                            f"\033[91mFailed job due to user error with checkpoint path:\033[0m"
                            f" {r.job_config.checkpoint_path}/{r.job_config.checkpoint_name}"
                        )

                        if self.output_error_messages:
                            print(f"Config: {r.job_config}")
                            print(f"\033[38;5;208mError: {r.exception}\033[0m")
                            print(f"\033[38;5;208mStack Trace:\033[0m\n{r.stack_trace}")


                # Either way put the job in the finished jobs as putting it back in the queue won't help
                self._remove_job(i, result)

            except (submitit.core.utils.FailedJobError, submitit.core.utils.UncompletedJobError) as e:

                if self.output_error_messages:
                    print(f"Job {current_job.job_config.checkpoint_path} was interrupted")
                    print(f"Error: {e}")
                # Requeue Logic
                if current_job.retries >= self.max_retries:
                    self._remove_job(i, FailedJobState(TimeoutError("too many retries"), "to_many_retries",
                                                       current_job.job_config, current_job.wandb_config))
                    print(f"\033[94mJob {current_job.job_config.checkpoint_path} failed after {self.max_retries} retries\033[0m")
                    if self.output_error_messages:
                        print(f"Error: {e}")
                    continue

                # Resubmit the job
                current_job.retries += 1
                current_job.job = self._executor.submit(
                    self._job_cls(current_job.job_config, current_job.wandb_config)
                )

        # If we ended up finishing a job, we can add a new one
        self._update_submitted_queue()

    def _remove_job(self, idx, result):
        """Private helper method to remove a job from the currently working
        on queue
        """
        job_book_keeping = self._running_jobs[idx]
        self._running_jobs[idx] = None
        job_book_keeping.result = result
        job_book_keeping.job = None
        self._finished_jobs.append(job_book_keeping)

        # Update the progress bar
        if self.progress_bar is not None:
            self.progress_bar.update(1)

    def _cancel_all(self):
        """Private helper method to cancel all running jobs"""
        for job in self._running_jobs:
            if job is not None:
                job.job.cancel(check=False)

    @property
    def tasks_left(self):
        """This method returns the number of tasks left

        Returns:
            int: The number of tasks left
        """
        return len(self._pending_jobs) + sum(1 for job in self._running_jobs if job is not None)

    def done(self):
        """This method returns if the entire task is finished

        Returns:
            bool: True if the task is finished, False otherwise
        """
        return len(self._pending_jobs) == 0 and all(job is None for job in self._running_jobs)

    @property
    def results(self):
        """
        Returns the results in the order in which they were completed. Note, if you
        have a multitask job there will be nested lists so it is list[list[job results]]. If
        just single task jobs, it will be a list[results] where results are what your job returns
        """
        return [job.result for job in self._finished_jobs]

    def __str__(self):
        result = ""
        for output in self.results:
            result += str(output) + "\n"
        return result
