#!/bin/bash
#SBATCH --gres=gpu:gtx:1
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --output=/media/compute/homes/dmindlin/medical-decision-support/slurm_logs/train_wrist_xray.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun train_wrist_xray.job.sh