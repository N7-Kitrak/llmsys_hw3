#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 24:00:00
#SBATCH --gpus=v100-32:2
#SBATCH --output=mt_output_%j.log
#SBATCH --error=mt_error_%j.log

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

# Load modules
module load cuda/12.4.0

# Go to your project directory
cd $PROJECT/llmsys_hw3

# Create virtual environment (only if not already created)
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python=3.12
fi

# Activate virtual environment
source .venv/bin/activate

# (Optional) install deps if needed
# uv pip install -r requirements.txt

nvidia-smi

# Run your script
python project/run_machine_translation.py

echo "Job finished on $(date)"