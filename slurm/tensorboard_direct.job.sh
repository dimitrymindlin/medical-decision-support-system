#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/medical-decision-support-system

tensorboard dev upload --logdir tensorboard_logs/logs_direct \
    --name "MDS final model without Pretraining " \
    --description "Final models on the mura dataset trained only on XR-WRIST" \
    --one_shot
