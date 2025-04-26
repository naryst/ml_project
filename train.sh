cd seed-vc

python train_v2.py \
--config configs/v2/vc_wrapper.yaml \
--dataset-dir ../dataset/source_voice \
--run-name russian_train_2 \
--batch-size 4 \
--max-steps 999999999 \
--max-epochs 1000 \
--save-every 2000 \
--num-workers 0 \
--train-cfm \
--train-ar