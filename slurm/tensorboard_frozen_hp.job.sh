#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/medical-decision-support-system

tensorboard dev upload --logdir tensorboard_logs/logs_frozen_hp \
    --name "Mura Wrist Last Layers HP Board" \
    --description "Comparison of several hyperparameters and pretraining" \
    --one_shot
