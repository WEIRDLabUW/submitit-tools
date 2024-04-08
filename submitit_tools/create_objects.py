from typing import Union

import submitit

from submitit_configs.run_config import WandbConfig
from submitit_configs.submitit_config import SubmititExecutorConfig
from dataclasses import asdict
import wandb


def create_executor(config: SubmititExecutorConfig):
    kwargs = asdict(config)

    executor = submitit.AutoExecutor(folder=kwargs.pop("root_folder"))

    executor.update_parameters(**kwargs)

    return executor


def init_wandb(config: Union[WandbConfig, None]):
    if config is None:
        return
    kwargs = asdict(config)
    wandb.init(**kwargs)
