#! /bin/bash

cd seed_vc_v2_minimal
python app_vc_v2.py \
--ar-checkpoint-path ../ru_checkpoints/models--narySt--voice_clonning/snapshots/5b12713180dc7bfd1888618e356ddab88c596546/AR_epoch_00000_step_17700.pth \
--cfm-checkpoint-path ../ru_checkpoints/models--narySt--voice_clonning/snapshots/5b12713180dc7bfd1888618e356ddab88c596546/CFM_epoch_00000_step_17700.pth \
--config configs/v2/vc_wrapper_gigaam.yaml
