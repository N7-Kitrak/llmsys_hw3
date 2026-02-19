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
cd llmsys_hw3

# create venv once
if [ ! -d ".venv" ]; then
  uv venv --python=3.12
fi
source .venv/bin/activate

# install deps once (or when requirements change)
if [ ! -f ".venv/.deps_installed" ]; then
  uv pip install -r requirements.extra.txt
  uv pip install -r requirements.txt
  uv pip install -e .
  touch .venv/.deps_installed
fi

# (Optional) install deps if needed
# uv pip install -r requirements.txt

nvidia-smi

# Run your script
python project/run_machine_translation.py

echo "Job finished on $(date)"
