from dataclasses import dataclass
from typing import Union


@dataclass
class BaseJobConfig:
    checkpoint_path: str = "checkpoints"
    checkpoint_name: str = "checkpoint.pt"

    def __post_init__(self):
        # Do something here that is smart for the parameters:
        pass


@dataclass
class WandbConfig:
    # Change these values to your liking
    project: str = "default_project_name"
    name: str = "default_run_name"
    tags: Union[str, list] = "default_tags"
    notes: str = "default_notes"
    resume: str = "allow"
    mode: str = "online"  # Wandb sometimes causes problems with the voulume of api calls so you can set this to "ofline"
    id: str = None  # This MUST be a unique id that can be used to resume the run otherwise you will 
                    # have weird conflicting wandb things going on. Submitit-tools will do this automatically