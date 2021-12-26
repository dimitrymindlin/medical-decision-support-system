#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/medical-decision-support-system

python3 -m mura_pretraining.scripts.train --augmentation --use_class_weights