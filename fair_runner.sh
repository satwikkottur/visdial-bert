#!/bin/bash
## SLURM scripts have a specific format. 

## job name
#SBATCH --job-name=cd_qi_8
#SBATCH --output=/checkpoint/%u/logs/visdial_bert/%x-%j.out
#SBATCH --error=/checkpoint/%u/logs/visdial_bert/%x-%j.err

## partition name
#SBATCH --partition=learnfair
## number of nodes
#SBATCH --nodes=1
#SBATCH --time=3600
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=10
## number of tasks per node
#SBATCH --ntasks-per-node=1

srun --label wrapper.sh
# srun --label wrapper_deep.sh
