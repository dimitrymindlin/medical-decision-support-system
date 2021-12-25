#!/bin/bash
#SBATCH --gres=gpu:gtx:1
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --output=/media/compute/homes/tmarkmann/example.out

srun example.job.sh