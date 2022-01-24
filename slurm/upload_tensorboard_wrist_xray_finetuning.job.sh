#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/medical-decision-support-system

tensorboard dev upload --logdir logs_wrist_xray_finetuning \
    --name "WristXray final Finetuning Board" \
    --description "Unfreeze and finetune the frozen trained WristXrayNet" \
    --one_shot
