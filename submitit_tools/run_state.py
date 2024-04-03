import submitit
from typing import List, Type, Union
from configs import BaseRunConfig, WandbConfig
from tqdm.auto import tqdm
from submitit_tools.base_classes import BaseJob, JobBookKeeping

FAILED_JOB = "FAILED_JOB"


class SubmititState:
    """
    This class is used to store the state of the running jobs
    """

    def __init__(self,
                 executor: submitit.AutoExecutor,
                 job_cls: Type[BaseJob],
                 job_run_configs: List[BaseRunConfig],
                 job_wandb_configs: List[Union[WandbConfig, None]],
                 with_progress_bar: bool = False,
                 max_retries: int = 30,
                 num_concurent_jobs: int = 10):

        self.executor: submitit.AutoExecutor = executor
        self.job_cls: Type[BaseJob] = job_cls
        self.max_retries: int = max_retries
        self.progress_bar = tqdm(total=len(job_run_configs), desc="Job Progress") if with_progress_bar else None

        self.pending_jobs: List[JobBookKeeping] = [
            JobBookKeeping(run_config, wandb_config) for run_config, wandb_config in
            zip(job_run_configs, job_wandb_configs)
        ]

        self.running_jobs: List[Union[JobBookKeeping, None]] = [None] * num_concurent_jobs

        self.finished_jobs: List[JobBookKeeping] = []

        self._update_submitted_queue()

    def _update_submitted_queue(self):
        if len(self.pending_jobs) == 0:
            # No pending jobs to add
            return
        for i in range(len(self.running_jobs)):
            if self.running_jobs[i] is not None:
                # We already have a running job or no pending left at this index so don't do anything
                continue
            # We have to queue a job if we can
            job = self.executor.submit(
                self.job_cls(self.pending_jobs[0].run_config, self.pending_jobs[0].wandb_config)
            )
            self.running_jobs[i] = self.pending_jobs.pop(0)
            self.running_jobs[i].job = job
            assert job is not None, "Job is None"
    def update_state(self):
        for i in range(len(self.running_jobs)):
            if self.running_jobs[i] is None:
                continue

            current_job = self.running_jobs[i]
            assert current_job.job is not None, "Job is running but job is None"

            if not current_job.done():
                continue

            # The job is finished! It might have failed:
            try:
                result = current_job.result()

                # If we get here, the job worked!
                self._remove_job(i, result)

            except (submitit.core.utils.FailedJobError, submitit.core.utils.UncompletedJobError) as e:

                # Requeue Logic
                if current_job.retries > self.max_retries:
                    self._remove_job(i, FAILED_JOB)
                    print(f"Job {current_job.run_config.checkpoint_path} failed after {self.max_retries} retries")
                    print(f"Error: {e}")
                    continue

                # Resubmit the job
                current_job.retries += 1
                current_job.job = self.executor.submit(
                    self.job_cls(current_job.run_config, current_job.wandb_config)
                )

        # If we ended up finishing a job, we can add a new one
        self._update_submitted_queue()

    def _remove_job(self, idx, result):
        job_book_keeping = self.running_jobs[idx]
        self.running_jobs[idx] = None
        job_book_keeping.result = result
        job_book_keeping.job = None
        self.finished_jobs.append(job_book_keeping)

        # Update the progress bar
        if self.progress_bar is not None:
            self.progress_bar.update(1)

    @property
    def tasks_left(self):
        """This method returns the number of tasks left

        Returns:
            int: The number of tasks left
        """
        return len(self.pending_jobs) + sum(1 for job in self.running_jobs if job is not None)

    def done(self):
        """This method returns if the entire task is finished

        Returns:
            bool: True if the task is finished, False otherwise
        """
        return len(self.pending_jobs) == 0 and all(job is None for job in self.running_jobs)

    @property
    def results(self):
        return [job.result for job in self.finished_jobs]

    def __str__(self):
        result = ""
        for output in self.results:
            result += str(output) + "\n"
        return result
