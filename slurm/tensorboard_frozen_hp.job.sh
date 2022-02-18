#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/medical-decision-support-system

tensorboard dev upload --logdir tensorboard_logs/logs_frozen_hp \
    --name "MDS Frozen HP" \
    --description "Comparison of several hyperparameters on the pretrained model for XR-WRIST training" \
    --one_shot
