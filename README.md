# submitit-tools


Notes:
If a job crashes, there is no way to distinguish this between being preempted or just crashed
which can probably lead to problems. 

We hypothisize that if a job is prempted, we have to handle the requing,
but if it is just timed out at 4 hours, submitit will manage it