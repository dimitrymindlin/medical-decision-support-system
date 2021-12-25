#!/bin/bash
source /media/compute/homes/dmindlin/miniconda3/etc/profile.d/conda.sh
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/medical-decision-support-system

python3 -m medical-decision-support-system.mura_pretraining.scripts.train