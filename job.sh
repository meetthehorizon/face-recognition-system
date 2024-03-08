#!/bin/bash
#SBATCH --job-name=distributed_training
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=distributed_training_%j.out
#SBATCH --error=distributed_training_%j.err

./home/apps/spack/share/spack/setup-env.sh
spack load cuda
spack load cuda@12.3.0

source /home/kshitij.cse22.itbhu/miniconda3/bin/activate
conda activate pytorch

PYTHON_SCRIPT="-m main config/config.yaml"

# Number of GPUs per node
GPUS_PER_NODE=2

# Calculate the total number of processes
TOTAL_PROCESSES=$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))

# Distributed training command
srun --mpi=pmix_v3 -N $SLURM_JOB_NUM_NODES -n $TOTAL_PROCESSES python $PYTHON_SCRIPT
