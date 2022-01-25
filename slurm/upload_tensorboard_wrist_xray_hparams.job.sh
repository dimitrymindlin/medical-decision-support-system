#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/medical-decision-support-system

tensorboard dev upload --logdir logs_wrist_xray_hparams \
    --name "WristXray Hparams Board" \
    --description "Hparams for densenet" \
    --one_shot
