import time

from submitit_tools.custom_jobs import SimpleAddJob
from submitit_tools.run_state import SubmititState
from submitit_configs import  ExampleRunConfig, SubmititExecutorConfig
from submitit_tools.create_objects import create_executor


run_configs = []
wandb_configs = []
sub_cfg = SubmititExecutorConfig()
sub_cfg.timeout_min = 4
sub_cfg.partition = "ckpt"


executor = create_executor(sub_cfg)


for i in range(10):
    run_configs.append(ExampleRunConfig(first_number=0, second_number=i))
    wandb_configs.append(None)

state = SubmititState(
    executor=executor,
    job_cls=SimpleAddJob,
    job_run_configs=run_configs,
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
