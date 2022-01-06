#!/bin/bash
#SBATCH -p cpu
#SBATCH -c2
#SBATCH --time=1:00:00
#SBATCH --output=/media/compute/homes/dmindlin/medical-decision-support-system/slurm_logs/tensorboard_mura_tuning.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1

srun upload_tensorboard_mura_tuning.job.sh