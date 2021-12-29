#!/bin/bash
#SBATCH --gres=gpu:gtx:1
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --output=/media/compute/homes/dmindlin/medical-decision-support-system/slurm_logs/train_mura.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun train_mura.job.sh