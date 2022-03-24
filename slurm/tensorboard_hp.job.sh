#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/medical-decision-support-system

tensorboard dev upload --logdir tensorboard_logs/logs_hp \
    --name "MDS Direct HP Board" \
    --description "Direct HP Board" \
    --one_shot

