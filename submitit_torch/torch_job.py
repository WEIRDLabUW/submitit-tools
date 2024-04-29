from dataclasses import asdict, dataclass
from typing import Any, Dict, Union
from collections import OrderedDict

import fsspec
import submitit
import torch
import wandb
import os

from .torch_config import TorchMultiprocessingJobConfig
from submitit_configs import WandbConfig
from submitit_tools import BaseJob, run_nvidia_smi
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    optimizer_state: Dict[str, Any]
    finished_epoch: int


class TorchJob(BaseJob):
    """
    This is a basic job that runs a distributed training loop using PyTorch.
    """

    def __init__(self, job_config: TorchMultiprocessingJobConfig, wandb_config: Union[WandbConfig, None]):
        # I need to handle wandb myself as I do not want every job to initalize a wandb run
        assert wandb_config is not None, "You must provide a wandb config"
        super().__init__(job_config, None)
        self.job_config: TorchMultiprocessingJobConfig = job_config
        self._wandb_config: WandbConfig = wandb_config

    def _initialize(self):
        dist_env = submitit.helpers.TorchDistributedEnvironment().export()

        print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        print(f"rank: {dist_env.rank}")
        print(f"world size: {dist_env.world_size}")
        print(f"local rank: {dist_env.local_rank}")
        print(f"local world size: {dist_env.local_world_size}")

        torch.distributed.init_process_group(backend="nccl")
        assert dist_env.rank == torch.distributed.get_rank()
        assert dist_env.world_size == torch.distributed.get_world_size()
        self.local_rank = dist_env.local_rank
        self.global_rank = dist_env.rank
        self.world_size = dist_env.world_size

        # Not implemented yet
        self.lr_scheduler = None

        model, optimizer, train_dataset, test_dataset = self.job_config.create_artifacts(self.job_config)
        self.model = model
        self.optimizer = optimizer
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = self._prepare_dataloader(test_dataset) if test_dataset is not None else None

        self.epochs_run = 0

        self.model = self.model.cuda()

        if self.job_config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        if self.checkpoint_exists():
            self.load_checkpoint()
        if self.job_config.compile:
            self.model = torch.compile(self.model)

        self.model = DDP(self.model).cuda()

        # initialize wandb:
        if self.global_rank == 0:
            wandb.init(**asdict(self._wandb_config))
            wandb.config.update(asdict(self.job_config))
        print(f"nvidia-smi output = {run_nvidia_smi()}")

    def _job_call(self):
        """
        This method is called by the call method and contains the entire job
        """
        for epoch in range(self.epochs_run, self.job_config.max_epochs):
            print(f"Running epoch {epoch}")
            epoch += 1
            avg_loss = self._run_epoch(epoch, self.train_loader, train=True)

            # eval run
            if self.test_loader:
                test_avg_loss = self._run_epoch(epoch, self.test_loader, train=False)
                test_loss = torch.tensor([test_avg_loss]).cuda()
                dist.reduce(test_loss, 0, dist.ReduceOp.SUM)
            self.epochs_run = epoch
            if self.global_rank == 0:
                print(f"In epoch {epoch} with rank {self.global_rank} the loss is {avg_loss}")
                log_dict = {"loss": avg_loss, "learning_rate": self.optimizer.param_groups[0]['lr']}
                if self.test_loader:
                    test_loss = test_loss / self.world_size
                    log_dict['test_loss'] = test_loss.item()
                    wandb.log(log_dict, step=epoch)
                else:
                    wandb.log(log_dict, step=epoch)

                if epoch % self.job_config.save_every == 0:

                    self._save_checkpoint()

                if self.lr_scheduler:
                    # assumes that this is the reduce on plateau one
                    self.lr_scheduler.step(metrics=avg_loss)
                print(f"Epoch {epoch} completed with rank {self.global_rank}")
        if self.global_rank == 0:
            # Run final things here
            pass

        return f"Final loss is {avg_loss}"

    def _run_batch(self, source, targets, train: bool = True) -> float:
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                                               enabled=self.job_config.use_amp):
            loss = self.job_config.get_loss(self.model, source, targets)

        if train:
            # Set to none is faster as it doesn't overwrite the memory
            self.optimizer.zero_grad(set_to_none=True)
            if self.job_config.use_amp:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.job_config.grad_norm_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.job_config.grad_norm_clip)
                self.optimizer.step()

        return loss.item()

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        dataloader.sampler.set_epoch(epoch)
        losses = []
        for source, targets in dataloader:
            source = source.cuda()
            targets = targets.cuda()
            batch_loss = self._run_batch(source, targets, train)
            losses.append(batch_loss)

        return sum(losses) / len(dataloader)

    def _save_checkpoint(self):
        """
        This is a helper method that you can use to save the state of your job. Make sure to call it
        periodically in your job incase it is killed and requeued. ONLY CALL THIS IN YOUR _job_call METHOD
        """
        model = self.model

        # This unwraps the state dict from both torch.compile() as well as DDP()
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_model = model.module if hasattr(model, "module") else raw_model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=self.epochs_run
        )

        # save snapshot
        snapshot = asdict(snapshot)
        torch.save(snapshot, self.save_path)

        print(f"Snapshot saved at epoch {self.epochs_run}")

    def load_checkpoint(self):
        snapshot = fsspec.open(self.save_path)
        with snapshot as f:
            snapshot_data = torch.load(f, map_location="cpu")

        snapshot = Snapshot(**snapshot_data)
        self.model.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epochs_run = snapshot.finished_epoch
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _prepare_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.job_config.batch_size,
            pin_memory=True,
            num_workers=self.job_config.data_loader_workers,
            sampler=DistributedSampler(dataset)
        )
