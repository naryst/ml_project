#! /bin/bash

cd seed_vc_v1_minimal

python inference.py \
--source ../dataset/test_wavs/bronya.wav \
--target ../dataset/test_wavs/bronya.wav \
--output ../dataset/bronya_converted.wav \
--config configs/sv_v1_small.yaml \
--checkpoint ../ru_checkpoints/russian_train_5/ft_model.pth
