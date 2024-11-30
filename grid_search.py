import os
import itertools

# Define the hyperparameter space
hyperparameter_space = {
    "learning_rate": [0.001],
    "step_size": [5, 10,15,20],
    "gamma": [0.9],
    "weight_decay": [0.0009],
    "epochs": [1],
}

# Generate all combinations of hyperparameters
param_combinations = list(itertools.product(
    hyperparameter_space["learning_rate"],
    hyperparameter_space["step_size"],
    hyperparameter_space["gamma"],
    hyperparameter_space["weight_decay"],
    hyperparameter_space["epochs"],
))

# Directory for task file and logs
job_dir = "slurm_jobs"
os.makedirs(job_dir, exist_ok=True)

# Create a task file with one line per parameter combination
task_file_path = os.path.join(job_dir, "tasks.txt")
with open(task_file_path, "w") as task_file:
    for params in param_combinations:
        learning_rate, step_size, gamma, weight_decay, epochs = params
        task_file.write(f"{learning_rate} {step_size} {gamma} {weight_decay} {epochs}\n")

# Create a single Slurm script for the job array
job_script_path = os.path.join(job_dir, "job_array.slurm")
with open(job_script_path, "w") as job_script:
    job_script.write(f"""#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH -t 0-00:20
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH -o {job_dir}/log_%A_%a.out
#SBATCH -e {job_dir}/log_%A_%a.err
#SBATCH --array=1-{len(param_combinations)}%3

# Ensure the job directory exists
mkdir -p {job_dir}

# Get the line corresponding to the current array task ID
TASK=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {task_file_path})

# Parse the task parameters
read learning_rate step_size gamma weight_decay epochs <<< $TASK


UNIQUE_ID="$SLURM_JOB_ID_$SLURM_ARRAY_TASK_ID"


# Run the Python training script
srun python3 train.py \\
    --learning_rate $learning_rate \\
    --step_size $step_size \\
    --gamma $gamma \\
    --weight_decay $weight_decay \\
    --epochs $epochs \\
    --job_id $UNIQUE_ID
""")

# Submit the job array
os.system(f"sbatch {job_script_path}")
print(f"Submitted job array with {len(param_combinations)} tasks.")