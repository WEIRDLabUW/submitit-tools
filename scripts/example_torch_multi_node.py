import torchvision
import wandb

from submitit_torch import TorchMultiprocessingJobConfig
from submitit_torch import TorchJob
from submitit_tools import SubmititState
from submitit_configs import WandbConfig, SubmititExecutorConfig

import torch
import torch.nn as nn


def get_loss(net, x, y):
    return torch.nn.functional.mse_loss(net(x), y)


def create_artifacts(config: TorchMultiprocessingJobConfig):
    model = torch.nn.Sequential(
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_dataset = torchvision.datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor()
    )
    return model, optimizer, train_dataset, None


def main():
    job_config = TorchMultiprocessingJobConfig(get_loss=get_loss, create_artifacts=create_artifacts, batch_size=64)
    wandb_config = WandbConfig(
        project="submitit-torch-test",
        tags=["torch", "mnist"],
        id=wandb.util.generate_id(),
    )
    executor_config = SubmititExecutorConfig(
        slurm_additional_parameters={},
        slurm_gpus_per_node=4,
        slurm_ntasks_per_node=4,
        slurm_name="submitit-test",
        timeout_min=60 * 2,
        cpus_per_task=8,
        mem_gb=64,
        slurm_partition="gpu-l40"
    )
    executor_config.__delattr__("slurm_gpus_per_task")

    print(executor_config)

    executor = SubmititState(
        TorchJob,
        executor_config=executor_config,
        job_run_configs=[job_config],
        job_wandb_configs=[wandb_config],
        with_progress_bar=True,
        output_error_messages=True
    )

    while executor.done() is False:
        executor.update_state()

    print(executor.results)


if __name__ == '__main__':
    main()
