#!/bin/bash
#SBATCH --gres=gpu:gtx:1
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --output=/media/compute/homes/dmindlin/load_wrist_xray.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun load_wrist_xray_dataset.job.sh