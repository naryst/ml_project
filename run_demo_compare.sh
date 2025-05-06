#!/bin/bash

cd seed_vc_v1_minimal

# Set default config paths
CONFIG_PATH="configs/sv_v1_small.yaml"

# Run the demo with default configs
python app_vc_compare.py \
    --config1 $CONFIG_PATH \
    --config2 $CONFIG_PATH \
    --fp16 True \
    --share True
