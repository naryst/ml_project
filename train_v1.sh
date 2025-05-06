cd seed_vc_v1_minimal

python train.py \
--config configs/sv_v1_small.yaml \
--dataset-dir ../dataset/source_voices \
--run-name russian_train_6 \
--batch-size 4 \
--max-steps 200 \
--max-epochs 1 \
--save-every 5000 \
--num-workers 0 \
--pretrained-ckpt ../ru_checkpoints/russian_train_5/ft_model.pth