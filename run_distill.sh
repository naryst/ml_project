#!/bin/bash

cd seed_vc_v1_minimal

python distillation.py \
    --teacher_config configs/conf_wisper.yaml \
    --student_config configs/conf_wisper.yaml \
    --data_dir ../dataset/source_voices \
    --output_dir ../distillation_results \
    --initial_teacher_steps 16 \
    --final_steps 2 \
    --steps_reduction_factor 2 \
    --epochs_per_iteration 1 \
    --batch_size 1 \
    --num_workers 0 \
    --save_interval 3000 \
    --use_trajectory_loss \
    --trajectory_weight_type exponential \
    --iterations_per_epoch 5000 \
    --eval_steps 2000 