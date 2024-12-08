import os
import itertools

# # Define the hyperparameter space
# hyperparameter_space = {
#     "learning_rate": [1,0.1,0.05],
#     "step_size": [1,5,10],
#     "gamma": [0.9,0.1],
#     "weight_decay": [0.0009],
#     "epochs": [15],
#     "batch_size":[8,16,32],
#     "real_size":[0,100,200,300,400,500,600,700,800,900,1000]
# }

hyperparameter_space = {
    "learning_rate": [0.001],
    "step_size": [3],
    "gamma": [0.9],
    "weight_decay": [0.001],
    "epochs": [10],
    "batch_size":[4],
    "real_size":[200,400,600,800,1000]
}



# Generate all combinations of hyperparameters
param_combinations = list(itertools.product(
    hyperparameter_space["learning_rate"],
    hyperparameter_space["step_size"],
    hyperparameter_space["gamma"],
    hyperparameter_space["weight_decay"],
    hyperparameter_space["epochs"],
    hyperparameter_space["batch_size"],
    hyperparameter_space["real_size"],

))

# Directory for task file and logs
job_dir = "slurm_jobs"
os.makedirs(job_dir, exist_ok=True)

# Create a task file with one line per parameter combination
task_file_path = os.path.join(job_dir, "tasks.txt")
with open(task_file_path, "w") as task_file:
    for params in param_combinations:
        learning_rate, step_size, gamma, weight_decay, epochs, batch_size , real_size= params
        task_file.write(f"{learning_rate} {step_size} {gamma} {weight_decay} {epochs} {batch_size} {real_size}\n")

# Create a single Slurm script for the job array
job_script_path = os.path.join(job_dir, "job_array.slurm")
with open(job_script_path, "w") as job_script:
    job_script.write(f"""#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH -t 0-01:20
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --array=1-{len(param_combinations)}%3
#SBATCH -o /cluster/home/oeiels01/Squash_L3D/Results/%A_%a.out
#SBATCH -e /cluster/home/oeiels01/Squash_L3D/Results/%A_%a.err

RESULTS_DIR="/cluster/home/oeiels01/Squash_L3D/Results"
mkdir -p $RESULTS_DIR

# Use SLURM_ARRAY_JOB_ID and SLURM_ARRAY_TASK_ID for consistent naming
UNIQUE_ID="${{RESULTS_DIR}}/${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
mkdir -p $UNIQUE_ID

echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

eval "$(micromamba shell hook bash)"
micromamba activate /cluster/tufts/cs152l3dclass/shared/micromamba/envs/l3d_2024f_cuda_readonly/
cd /cluster/home/oeiels01/Squash_L3D

# Retrieve parameters for this task
TASK=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {task_file_path})
read learning_rate step_size gamma weight_decay epochs batch_size real_size <<< $TASK

python3 generate_ratio_csv.py \\
      --end_row $real_size\\
      --output_file $UNIQUE_ID/train.csv

# Run your training
srun python3 train.py \\
    --train_csv $UNIQUE_ID/train.csv \\
    --learning_rate $learning_rate \\
    --step_size $step_size \\
    --gamma $gamma \\
    --weight_decay $weight_decay \\
    --epochs $epochs \\
    --job_id $UNIQUE_ID \\
    --train_batch_size $batch_size \\
    --verbose

    
# # Remove all .pth files after training is complete  
# find $UNIQUE_ID -name "*.pth" -type f -delete


# Move logs into the UNIQUE_ID directory
mv $RESULTS_DIR/${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.out $UNIQUE_ID/log_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.out
mv $RESULTS_DIR/${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.err $UNIQUE_ID/log_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.err
""")




# Submit the job array
os.system(f"sbatch {job_script_path}")
print(f"Submitted job array with {len(param_combinations)} tasks.")