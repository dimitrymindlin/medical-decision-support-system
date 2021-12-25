#!/bin/bash
source /media/compute/homes/tmarkmann/miniconda3/etc/profile.d/conda.sh
conda activate chexnet
cd /media/compute/homes/tmarkmann/chexnet-tf2

python3 -m chexnet.scripts.xai