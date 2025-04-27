cd seed-vc_minimal

python train_v2.py \
--config configs/v2/vc_wrapper.yaml \
--dataset-dir ../dataset/source_voices \
--run-name russian_train_4 \
--batch-size 8 \
--max-steps 999999999 \
--max-epochs 1 \
--save-every 10000 \
--num-workers 0 \
--train-cfm \
--train-ar
