TASK_NAME='qg_dutch'
MODEL_NAME='flax-community/t5-base-dutch'
EX_TIME=$(date +"%Y-%h-%d-(%H:%M)")

echo "=============RUN EXP=============!"
echo 'train->dutch -*- eval->dutch'
echo $TASK_NAME
echo $EX_TIME
echo "=============RUN EXP=============!"
python3 pretrain_qg.py \
  --task $TASK_NAME \
  --model_name_or_path $MODEL_NAME \
  --logging_dir "runs/${TASK_NAME}_dutch_t5_${EX_TIME}/logs" \
  --output_dir "runs/${TASK_NAME}_dutch_t5_${EX_TIME}" \
  --cache_dir "runs/cache/" \
  --train_file_path "raw_data/qg_dutch.json" \
  --valid_file_path "raw_data/qg_dutch.json" \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 6.25e-5 \
  --num_train_epochs 10 \
  --seed 42 \
  --do_train \
  --do_eval \
  --do_predict \
  --logging_steps 10 \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --overwrite_output_dir \
  --predict_with_generate \
  --is_debug_mode -1 || exit
