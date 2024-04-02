from submitit_tools.base_classes import BaseJob
import time

class MyJob(BaseJob):
    def __init__(self, run_config, wandb_config):
        self.run_config = run_config
        self.wandb_config = wandb_config

    def __call__(self):
        time.sleep(5)
        return self.run_config.first_number + self.run_config.second_number


    def checkpoint(self):
        pass

