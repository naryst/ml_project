#! /bin/bash

cd seed_vc_minimal

python inference_v2.py \
--source ../dataset/test_wavs/bronya.wav \
--target ../dataset/test_wavs/bronya.wav \
--output ../dataset/bronya_converted.wav \
--config configs/v2/vc_wrapper_gigaam.yaml
