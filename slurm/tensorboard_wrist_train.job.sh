#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/medical-decision-support-system

tensorboard dev upload --logdir tensorboard_logs/logs_frozen \
    --name "MDS Frozen" \
    --description "Training last layers the pretrained model of XR-Wrist" \
    --one_shot
