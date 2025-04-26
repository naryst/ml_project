#!/bin/bash

# Get current timestamp for log file name
LOG_FILE="eval.log"

cd seed-vc

# Run the evaluation and redirect both stdout and stderr to the log file
python eval.py \
--source ../dataset/source_voice \
--target ../dataset/target_voice \
--output eval_results \
--max-samples 100 \
--xvector-extractor wavlm 2>&1 | tee "$LOG_FILE"

echo "Evaluation completed. Log file: $LOG_FILE"
