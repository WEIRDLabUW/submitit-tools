from dataclasses import dataclass
from submitit_configs import BaseJobConfig
from typing import Callable, Tuple, Union
import torch
import torch.nn as nn
import torch.utils.data as data


@dataclass
class TorchMultiprocessingJobConfig(BaseJobConfig):
    # This is a function that takes this config, and should return a model to optimize,
    # a optimizer, and the train and optionally test dataset.
    create_artifacts: Callable[['TorchMultiprocessingJobConfig'],
    Tuple[nn.Module, torch.optim.Optimizer, data.Dataset, Union[data.Dataset, None]]] = None

    # This is a function that should return a loss tensor that .backward() can be called on. It's inputs
    # are the model, and the input, target from the dataloader. If you want to put something on to gpu, you MUST
    # use the .cuda() method.
    get_loss: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor] = None  # A function that returns the loss

    batch_size: int = 32
    shuffle: bool = True
    data_loader_workers: int = 4
    save_every: int = 1
    use_amp: bool = False
    grad_norm_clip: float = 1.0
    max_epochs: int = 4
    compile: bool = True
