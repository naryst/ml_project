cd seed_vc_v1_minimal

python train.py \
--config configs/sv_v1_small.yaml \
--dataset-dir ../dataset/source_voices \
--run-name russian_train_5 \
--batch-size 10 \
--max-steps 999999999 \
--max-epochs 1 \
--save-every 5000 \
--num-workers 0