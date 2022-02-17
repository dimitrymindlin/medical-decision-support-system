#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/medical-decision-support-system

tensorboard dev upload --logdir tensorboard_logs/logs_pretrain \
    --name "MDS Pretrain" \
    --description "Pretraining models on the mura dataset except of XR-WRIST" \
    --one_shot
