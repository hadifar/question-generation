#!/bin/bash

if test -z "$1"
then
      TASK_NAME="openstax_qg_simple"
else
      TASK_NAME=$1
fi

echo "=============training=============!"

python3 pretrain_qg.py \
    --model_name_or_path "t5-small" \
    --model_type "t5" \
    --logging_dir "runs/${TASK_NAME}/logs" \
    --output_dir "runs/${TASK_NAME}" \
    --train_file_path "raw_data/openstax_train.dataset.json" \
    --valid_file_path "raw_data/openstax_dev.dataset.json" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 6.25e-5 \
    --num_train_epochs 10 \
    --seed 42 \
    --do_train \
    --do_eval \
    --logging_steps 10 \
    --evaluation_strategy "epoch" \
    --overwrite_output_dir \
    --is_debug_mode -1 || exit

echo "=============evaluation=============!"

python3 eval.py \
    --model_name_or_path "runs/${TASK_NAME}" \
    --model_type "t5" \
    --tokenizer_name_or_path "runs/${TASK_NAME}" \
    --valid_file_path "raw_data/openstax_dev.dataset.json" \
    --is_debug_mode -1
