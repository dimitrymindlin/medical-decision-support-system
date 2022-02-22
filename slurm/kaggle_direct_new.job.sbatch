#!/bin/bash
#SBATCH --gres=gpu:tesla:2
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --output=/media/compute/homes/dmindlin/medical-decision-support-system/slurm_logs/kaggle_direct_new.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun kaggle_direct_new.job.sh