#!/bin/bash
#SBATCH --gres=gpu:gtx:1
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --output=/media/compute/homes/dmindlin/medical-decision-support-system/slurm_logs/reset_mura.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun resnet_mura_train.job.sh