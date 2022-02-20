#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/medical-decision-support-system

tensorboard dev upload --logdir tensorboard_logs/logs_finetune_hp \
    --name "MDS Finetuning Board" \
    --description "Final Finetuning of all layers on XR-Wrist" \
    --one_shot

