#!/bin/bash
#SBATCH -p cpu
#SBATCH -c2
#SBATCH --time=1:00:00
#SBATCH --output=/media/compute/homes/dmindlin/medical-decision-support-system/slurm_logs/evaluate_model.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun evaluate_model.job.sh