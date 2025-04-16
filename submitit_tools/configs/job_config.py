from dataclasses import dataclass
from typing import Union
import os
import pickle

@dataclass
class BaseJobConfig:
    checkpoint_path: str = "checkpoints"
    checkpoint_name: str = "checkpoint.pt"

    def __post_init__(self):
        # Do something here that uses passed paramaters to init other ones:
        pass
    
    def __str__(self):
        # a string method that prints it out in a more readable manner
        contents = "\n".join(f"\t{key}: {value}" for key, value in self.__dict__.items())
        return "{\n" + contents + "\n}"

    def get_job_id(self):
        # This will return the job id of the job if running on node.
        id =  os.environ.get("SLURM_JOB_ID", None)
        array_pos = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
        id = f"{id}_{array_pos}" 
        return id
    
    def mark_as_success(self, log_dir_path):
        # This will mark the job as successful. It will create a file in the log_dir_path
        # called success.txt. This is useful for debugging and logging.
        job_id = self.get_job_id()
        file_name = f"{job_id}_result.pkl"
        output = ('success', None)
        pickle.dump(output, open(os.path.join(log_dir_path, file_name), "wb"))

@dataclass
class WandbConfig:
    # Change these values to your liking
    project: str = "default_project_name" # this is the project for the run
    name: str = "default_run_name" # this is the run name
    tags: Union[str, list] = "default_tags"  # This is the tags that show up on wandb. Useful for sorting
    notes: str = "default_notes" # this is the notes that are used as a description field
    resume: str = "allow"
    mode: str = "online"  # Wandb sometimes causes problems with the volume of api calls so you can set this to "offline"
    id: str = None  # Submitit Tools will populate this automatically. Note: this exists because in order to resume, the runs
                    # must have a unique id.