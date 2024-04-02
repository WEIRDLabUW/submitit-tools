import time

from scripts.run_script import MyJob
from submitit_tools.run_state import SubmititState
from configs.run_config import ExampleRunConfig, WandbConfig
from configs.submitit_config import SubmititExecutorConfig
from submitit_tools.create_executor import create_executor


run_configs = []
wandb_configs = []
sub_cfg = SubmititExecutorConfig(
    timeout_min=4,
    partition="ckpt",
    slurm_array_parallelism=4,
)

executor = create_executor(sub_cfg)


for i in range(10):
    run_configs.append(ExampleRunConfig(first_number=0, second_number=0))
    wandb_configs.append(None)

state = SubmititState(
    executor=executor,
    job_cls=MyJob,
    job_run_configs=run_configs,
    job_wandb_configs=wandb_configs,
    with_progress_bar=True,
)

while state.done() is False:
    state.update_state()
    time.sleep(1)
