import submitit
from typing import List, Type, Union
from configs.run_config import BaseRunConfig, WandbConfig
from tqdm.auto import tqdm
from submitit_tools.base_classes import BaseJob

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
                 max_retries: int = 30):
        jobs = self._submit_job_list(executor, job_run_configs, job_wandb_configs, job_cls)

        self.executor: submitit.AutoExecutor = executor
        self.job_cls: Type[BaseJob] = job_cls
        self.max_retries: int = max_retries
        self.progress_bar = tqdm(total=len(jobs), desc="Job Progress") if with_progress_bar else None

        # In progress jobs
        self.jobs: List[submitit.Job] = jobs
        self.job_run_configs: List[BaseRunConfig] = job_run_configs
        self.job_wandb_configs: List[WandbConfig] = job_wandb_configs
        self.retries = [0 for _ in range(len(jobs))]

        # For completed jobs
        self.results = []
        self.finished_run_configs: List[BaseRunConfig] = []
        self.finished_wandb_configs: List[WandbConfig] = []

    def _submit_job_list(self,
                         executor: submitit.AutoExecutor,
                         run_config_list: List[BaseRunConfig],
                         wandb_config_list: List[WandbConfig],
                         job_cls: Type[BaseJob],
                         ) -> List[submitit.Job]:
        assert len(run_config_list) == len(
            wandb_config_list), "The length of the run_config_list and the wandb_config_list must be the same"
        job_results = []
        with executor.batch():
            for run_config, wandb_config in zip(run_config_list, wandb_config_list):
                job = executor.submit(job_cls(run_config, wandb_config))
                job_results.append(job)
        return job_results

    def update_state(self):
        for i in reversed(range(len(self.jobs))):
            job = self.jobs[i]
            if not job.done():
                continue

            # The job is finished! It might have failed:
            try:
                result = job.result()

                # If we get here, the job worked!
                self._remove_job(i, result)

            except (submitit.core.utils.FailedJobError, submitit.core.utils.UncompletedJobError) as e:

                # Requeue Logic
                if self.retries[i] > self.max_retries:
                    self._remove_job(i, FAILED_JOB)
                    print(f"Job {self.job_run_configs[i].checkpoint_path} failed after {self.max_retries} retries")
                    print(f"Error: {e}")
                    continue

                # TODO: I do not know how to requeue it within the scope of the executor
                # I will just resubmit it
                self.retries[i] += 1
                self.jobs[i] = self.executor.submit(self.job_cls(self.job_run_configs[i], self.job_wandb_configs[i]))

    def _remove_job(self, idx, result):
        job_name = self.job_run_configs[idx].checkpoint_path
        self.results.append(result)
        self.finished_run_configs.append(self.job_run_configs.pop(idx))
        self.finished_wandb_configs.append(self.job_wandb_configs.pop(idx))

        # Remove the job from the lists
        self.retries.pop(idx)
        self.jobs.pop(idx)

        # Update the progress bar
        if self.progress_bar is not None:
            self.progress_bar.update(1)
            # if result == FAILED_JOB:
            #     self.progress_bar.set_postfix(f"Failed job at {job_name}")
            # else:
            #     self.progress_bar.set_postfix(f"Finished job at {job_name}")

    def tasks_left(self):
        """This method returns the number of tasks left

        Returns:
            int: The number of tasks left
        """
        return len(self.jobs)

    def done(self):
        """This method returns if the entire task is finished

        Returns:
            bool: True if the task is finished, False otherwise
        """
        if len(self.jobs) == 0:
            return True
        return False

    def __str__(self):
        result = ""
        for output in self.results:
            result += str(output) + "\n"
        return result
