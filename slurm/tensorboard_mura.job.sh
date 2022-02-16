#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/medical-decision-support-system

tensorboard dev upload --logdir logs_mura \
    --name "Mura Pretraining" \
    --description "Pretraining models on the mura dataset except of Wrist Xrays" \
    --one_shot
