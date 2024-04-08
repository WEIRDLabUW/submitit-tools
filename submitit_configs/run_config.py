from dataclasses import dataclass
from typing import Union


@dataclass
class BaseRunConfig:
    checkpoint_path: str = "checkpoints"
    checkpoint_name: str = "checkpoint.pt"

    def __post_init__(self):
        # Do something here that is smart for the paramaters:
        pass
    
 
@dataclass
class WandbConfig:
    # Change these values to your liking
    project: str = "default_project_name"
    name: str = "default_run_name"
    tags: Union[str, list] = "default_tags"
    notes: str = "default_notes"
    resume: str = "allow"
    id: str = None   # You MUST overide this value with a unique id that can be used to resume the run
