TASK_NAME='qg'
MODEL_NAME='mrm8488/t5-base-finetuned-question-generation-ap'
EX_TIME=$(date +"%Y-%h-%d-(%H:%M)")

echo "=============RUN EXP=============!"
echo 'train->squad -*- eval->openqg'
echo $TASK_NAME
echo $EX_TIME
echo "=============RUN EXP=============!"
python3 pretrain_qg.py \
  --task $TASK_NAME \
  --model_name_or_path $MODEL_NAME \
  --logging_dir "runs/${TASK_NAME}_mrm8488_${EX_TIME}/logs" \
  --output_dir "runs/${TASK_NAME}_mrm8488_${EX_TIME}" \
  --train_file_path "raw_data/qg_train.json" \
  --valid_file_path "raw_data/qg_valid.json" \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate 6.25e-5 \
  --num_train_epochs 10 \
  --seed 42 \
  --do_eval \
  --do_predict \
  --logging_steps 10 \
  --evaluation_strategy "epoch" \
  --overwrite_output_dir \
  --predict_with_generate \
  --is_debug_mode -1 || exit

########################################################################################################################
########################################################################################################################
TASK_NAME='qg'
MODEL_NAME='mrm8488/t5-base-finetuned-question-generation-ap'
EX_TIME=$(date +"%Y-%h-%d-(%H:%M)")

echo "=============RUN EXP=============!"
echo 'train->squad-->openqg -*- eval->openqg'
echo $TASK_NAME
echo $EX_TIME
echo "=============RUN EXP=============!"
python3 pretrain_qg.py \
  --task $TASK_NAME \
  --model_name_or_path $MODEL_NAME \
  --logging_dir "runs/${TASK_NAME}_mrm8488_openqg_${EX_TIME}/logs" \
  --output_dir "runs/${TASK_NAME}_mrm8488_openqg_${EX_TIME}" \
  --train_file_path "raw_data/qg_train.json" \
  --valid_file_path "raw_data/qg_valid.json" \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate 6.25e-5 \
  --num_train_epochs 10 \
  --seed 42 \
  --do_train \
  --do_eval \
  --do_predict \
  --logging_steps 10 \
  --evaluation_strategy "epoch" \
  --overwrite_output_dir \
  --predict_with_generate \
  --is_debug_mode -1 || exit


########################################################################################################################
########################################################################################################################
TASK_NAME='qg'
MODEL_NAME='t5-base'
EX_TIME=$(date +"%Y-%h-%d-(%H:%M)")

echo "=============RUN EXP=============!"
echo 'train->openqg -- eval->openqg'
echo $TASK_NAME
echo $EX_TIME
echo "=============RUN EXP=============!"
python3 pretrain_qg.py \
  --task $TASK_NAME \
  --model_name_or_path $MODEL_NAME \
  --logging_dir "runs/${TASK_NAME}_openqg_t5_${EX_TIME}/logs" \
  --output_dir "runs/${TASK_NAME}_openqg_t5_${EX_TIME}" \
  --train_file_path "raw_data/qg_train.json" \
  --valid_file_path "raw_data/qg_valid.json" \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate 6.25e-5 \
  --num_train_epochs 10 \
  --seed 42 \
  --do_train \
  --do_eval \
  --do_predict \
  --logging_steps 10 \
  --evaluation_strategy "epoch" \
  --overwrite_output_dir \
  --predict_with_generate \
  --is_debug_mode -1 || exit

