import submitit
from configs.submitit_config import SubmititExecutorConfig
from dataclasses import asdict

def create_executor(config: SubmititExecutorConfig):
    kwargs = asdict(config)

    executor = submitit.AutoExecutor(folder=kwargs.pop("root_folder"))

    executor.update_parameters(**kwargs)
    
    return executor