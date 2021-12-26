#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/medical-decision-support-system

python3 -m medical-decision-support-system.mura_pretraining.dataloader.load_dataset