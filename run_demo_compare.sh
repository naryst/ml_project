#!/bin/bash

cd seed_vc_v1_minimal

# Set default config paths
CONFIG1="configs/conf_wisper.yaml"
CONFIG2="configs/conf_gigaam.yaml"

MODEL2_CHECKPOINT="../ru_checkpoints/russian_train_7/russian_train_7/ft_model.pth"

# Run the demo with default configs
python app_vc_compare.py \
    --config1 $CONFIG1 \
    --config2 $CONFIG2 \
    --fp16 True \
    --share True \
    --checkpoint2 $MODEL2_CHECKPOINT
