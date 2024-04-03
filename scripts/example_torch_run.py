import time
from configs import SubmititExecutorConfig, ExampleMNESTConfig, WandbConfig
from submitit_tools.create_objects import create_executor
from submitit_tools.create_objects import create_executor
from submitit_tools.run_state import SubmititState
from submitit_tools.custom_jobs import ExampleMNestJob

def generate_train_configs():
    wandb_configs = []
    job_configs = []
    for learning_rate in [0.001, 0.01, 0.1]:
        for batch_size in [32, 64, 128]:
            job_configs.append(ExampleMNESTConfig(
                learning_rate=learning_rate,
                batch_size=batch_size,
                checkpoint_name=f"lr_{learning_rate}_bs_{batch_size}.pt",
                checkpoint_path="mnest_checkpoints"
            ))
            wandb_configs.append(WandbConfig(
                project="mnest_test",
                name=f"lr_{learning_rate}_bs_{batch_size}",
                tags=["mnest", "test"],
                notes="This is a test run",
                resume="allow",
                id=f"lr_{learning_rate}_bs_{batch_size}"
            ))
    return job_configs, wandb_configs


def make_executor():
    config = SubmititExecutorConfig(root_folder="mnest_submitit_logs",
                                    slurm_name="submitit-test",
                                    timeout_min=60*2,
                                    cpus_per_task=16,
                                    mem_gb=24)
    return create_executor(config)

def main():
    executor = make_executor()
    job_configs, wandb_configs = generate_train_configs()
    state = SubmititState(
        executor=executor,
        job_cls=ExampleMNestJob,
        job_run_configs=job_configs,
        job_wandb_configs=wandb_configs,
        with_progress_bar=True,
        max_retries=1,
        num_concurent_jobs=4
    )
    while state.done() is False:
        state.update_state()
        time.sleep(1)

    for result in state.results:
        print(result)
