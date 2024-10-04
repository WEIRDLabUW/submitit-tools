import subprocess

def run_nvidia_smi():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')
