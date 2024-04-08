import os.path
import time
from dataclasses import asdict

import torch
import torch.nn as nn
import torchvision
import wandb

from submitit_configs import *
from submitit_tools.base_classes import BaseJob


class SimpleAddJob(BaseJob):
    def __init__(self, run_config: ExampleRunConfig, wandb_config: WandbConfig):
        self.run_config = run_config
        self.wandb_config = wandb_config

    def __call__(self):
        time.sleep(5)
        return self.run_config.first_number + self.run_config.second_number


class ExampleMNestJob(BaseJob):
    def __init__(self, run_config: ExampleMNESTConfig, wandb_config: WandbConfig):
        super().__init__(run_config, wandb_config)
        # Not needed, but helps with typing in pycharm
        self.run_config: ExampleMNESTConfig = run_config

        assert WandbConfig is not None, "This Job uses Wandb"
        dataset = torchvision.datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=run_config.batch_size,
            shuffle=True
        )
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
        )
        self.optimizer = torch.optim.Adam(self.network.parameters())

        if os.path.exists(os.path.join(run_config.checkpoint_path, run_config.checkpoint_name)):
            state_dict = torch.load(os.path.join(run_config.checkpoint_path, run_config.checkpoint_name),
                                    map_location=torch.device("cpu"))
            self.completed_epochs = state_dict["completed_epochs"]
            self.network.load_state_dict(state_dict["network"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
        else:
            self.completed_epochs = 0

    def __call__(self):
        super().__call__()
        self.network.to("cuda")

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
            wandb.log({"epoch_loss": epoch_loss})
            self.completed_epochs = epoch
            self._save_checkpoint()
        return f"Success! Paramaters: {asdict(self.run_config)}"

    def _save_checkpoint(self):
        state_dict = {
            "completed_epochs": self.completed_epochs,
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(state_dict, os.path.join(self.run_config.checkpoint_path, self.run_config.checkpoint_name))
