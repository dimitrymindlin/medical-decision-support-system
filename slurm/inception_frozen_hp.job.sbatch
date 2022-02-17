#!/bin/bash
#SBATCH --gres=gpu:gtx:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --output=/media/compute/homes/dmindlin/medical-decision-support-system/slurm_logs/inception_wrist_hp.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun inception_wrist_hp.job.sh