#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system
cd /media/compute/homes/dmindlin/medical-decision-support-system

python3 -m main.evaluate_model --direct --2022-08-08--10.09