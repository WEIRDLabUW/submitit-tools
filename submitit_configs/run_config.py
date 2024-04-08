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
    

    

# This is an example of a run config that you can use to run a simple addition task and log the output
@dataclass
class ExampleRunConfig(BaseRunConfig):
    first_number: int = 1
    second_number: int = 2
    
    def __post_init__(self):
        self.checkpoint_path = f"add_{self.first_number}_to_{self.second_number}"

@dataclass
class ExampleMNESTConfig(BaseRunConfig):
    learning_rate: float = 0.001
    num_epochs: int = 4
    batch_size: int = 32