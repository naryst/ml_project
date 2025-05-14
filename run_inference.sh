#! /bin/bash

cd seed_vc_v1_minimal

for i in {1..5}
do
  python inference.py \
  --source ../dataset/evaluation_samples/s${i}.wav \
  --target ../dataset/evaluation_samples/ref.wav \
  --output ../dataset/teacher_16steps \
  --config configs/conf_wisper.yaml \
  --diffusion-steps 16 
done