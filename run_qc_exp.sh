for TASK_NAME in 'cloze2normal' 'normal2cloze' 'multi_cloze2normal' 'multi_normal2cloze'; do

  EX_TIME=$(date +"%Y-%h-%d-(%H:%M)")
  echo "=============RUN EXP=============!"
  echo $TASK_NAME
  echo $EX_TIME
  echo "=============RUN EXP=============!"

  python3 pretrain_qg.py \
    --task $TASK_NAME \
    --model_name_or_path "t5-base" \
    --logging_dir "runs/${TASK_NAME}_${EX_TIME}/logs" \
    --output_dir "runs/${TASK_NAME}_${EX_TIME}" \
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
    --save_strategy "no" \
    --overwrite_output_dir \
    --predict_with_generate \
    --is_debug_mode 1 || exit

done

