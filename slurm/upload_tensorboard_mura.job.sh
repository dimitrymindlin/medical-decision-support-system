#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/medical-decision-support-system

tensorboard dev upload --logdir logs_mura \
    --name "Mura Pretraining Board" \
    --description "Comparison of several hyperparameters and pretraining" \
    --one_shot
