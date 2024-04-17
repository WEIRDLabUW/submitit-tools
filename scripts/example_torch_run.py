import time
from submitit_configs import SubmititExecutorConfig, WandbConfig
from submitit_tools import SubmititState
import os
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torchvision

from submitit_configs import BaseJobConfig, WandbConfig
from submitit_tools import BaseJob
import wandb


# This is an example of a run config that you can use to run a simple addition task and log the output

@dataclass
class MNISTRunConfig(BaseJobConfig):
    learning_rate: float = 0.001
    num_epochs: int = 4
    batch_size: int = 32


class ExampleMNISTJob(BaseJob):
    def __init__(self, run_config: MNISTRunConfig, wandb_config: WandbConfig):
        super().__init__(run_config, wandb_config)

        # Not needed, but helps with typing in pycharm
        self.run_config: MNISTRunConfig = run_config
        assert WandbConfig is not None, "This Job uses Wandb"

    def _initialize(self):
        # Load the data:
        dataset = torchvision.datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.run_config.batch_size,
            shuffle=True
        )
        self.completed_epochs = 0
        self.network = torch.nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.Softmax()
        ).to('cuda')
        self.optimizer = torch.optim.Adam(self.network.parameters())

        if os.path.exists(os.path.join(self.run_config.checkpoint_path, self.run_config.checkpoint_name)):
            checkpoint = torch.load(os.path.join(self.run_config.checkpoint_path, self.run_config.checkpoint_name))
            self.completed_epochs = checkpoint["completed_epochs"]
            self.network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    def __call__(self):
        # The super call loads wandb and calls the _initalize method
        super().__call__()

        # Run a standard training script
        for epoch in range(self.completed_epochs, self.run_config.num_epochs):
            epoch_loss = 0
            for data, target in self.data_loader:
                data, target = data.to("cuda"), target.to("cuda")
                self.optimizer.zero_grad()
                output = self.network(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(self.data_loader)

            # Important: You need to add the step flag
            wandb.log({"epoch_loss": epoch_loss}, step=epoch)
            self.completed_epochs = epoch
            # NOTE: You must do the checkpoint regularly.
            self._save_checkpoint()
        return f"Success! Paramaters: {asdict(self.run_config)}"

    def _save_checkpoint(self):
        # So that we do not overide a real checkpoint with a random init model. Not needed if you call
        # only after the __call__ method has been called.
        if not self.initialized:
            return
        # Save the checkpoing.
        state_dict = {
            "completed_epochs": self.completed_epochs,
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(state_dict, os.path.join(self.run_config.checkpoint_path, self.run_config.checkpoint_name))


def generate_train_configs():
    wandb_configs = []
    job_configs = []
    for learning_rate in [0.001, 0.01, 0.1]:
        for batch_size in [32, 64, 128]:
            job_configs.append(MNISTRunConfig(
                learning_rate=learning_rate,
                batch_size=batch_size,
                checkpoint_name=f"lr_{learning_rate}_bs_{batch_size}.pt",
                checkpoint_path="mnist_checkpoints"
            ))
            wandb_configs.append(WandbConfig(
                project="mnist_test",
                name=f"lr_{learning_rate}_bs_{batch_size}",
                tags=["mnist", "test"],
                notes="This is a test run",
                resume="allow",
                id=wandb.util.generate_id()
            ))
    return job_configs, wandb_configs




def main():
    config = SubmititExecutorConfig(root_folder="mnest_submitit_logs",
                                    slurm_name="submitit-test",
                                    timeout_min=60 * 2,
                                    cpus_per_task=16,
                                    mem_gb=24)
    job_configs, wandb_configs = generate_train_configs()
    state = SubmititState(
        job_cls=ExampleMNISTJob,
        executor_config=config,
        job_run_configs=job_configs,
        job_wandb_configs=wandb_configs,
        with_progress_bar=True,
        max_retries=4,
        num_concurent_jobs=4
    )

    while state.done() is False:
        state.update_state()
        time.sleep(1)

    for result in state.results:
        print(result)


if __name__ == "__main__":
    main()
