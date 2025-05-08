#!/bin/bash

cd seed_vc_v1_minimal

python distillation.py \
    --initial_teacher_checkpoint ../ru_checkpoints/russian_train_7/russian_train_7/ft_model.pth \
    --teacher_config configs/conf_gigaam.yaml \
    --student_config configs/conf_gigaam.yaml \
    --data_dir ../dataset/source_voices \
    --output_dir ../distillation_results \
    --initial_teacher_steps 32 \
    --final_steps 8 \
    --steps_reduction_factor 2 \
    --epochs_per_iteration 1 \
    --batch_size 1 \
    --num_workers 0 \
    --save_interval 1000 \
    --use_trajectory_loss \
    --trajectory_weight_type exponential \
    --device mps