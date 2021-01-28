#!/usr/bin/env bash
echo "=============Hello guys=============!"

PROJECT_NAME=${project_name}
PROJECT_PATHS=/project/users/amir/projects
#PROJECT_SCRATCH_PATHS=/project_scratch/users/amir/projects

export PATH=/project/users/amir/miniconda/conda/bin:$PATH
export PATH=$PROJECT_PATHS/$PROJECT_NAME:$PATH

echo "=============python path set=============!"
python --version # 3.7.3

echo "directory ..."
cd $PROJECT_PATHS/$PROJECT_NAME || exit
pwd

echo "=============install requirements=============!"
#apt-get update && apt-get install -y rsync && apt-get install -y libffi6



python3 -m pip install -r requirements.txt || exit


echo "=============installation done!!!=============!"


echo "=============preprocess=============!"

#python3 prepare_data.py || exit

echo "=============RUN QG=============!"

#python3 run_qg.py  || exit
python3 run_qg.py \
    --model_name_or_path ${model_name_or_path} \
    --model_type ${model_type} \
    --tokenizer_name_or_path ${tokenizer_name_or_path} \
    --output_dir ${output_dir} \
    --train_file_path ${train_file_path} \
    --valid_file_path ${valid_file_path} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} \
    --num_train_epochs ${num_train_epochs} \
    --seed 42 \
    --do_train \
    --do_eval \
    --logging_steps 100 \


#echo "=============training=============!"
#python3 -m torch.distributed.launch --nproc_per_node=${gpus} train.py --dataset_path=${dataset_name} --train_batch_size=${train_batch_size} --valid_batch_size=${valid_batch_size} --gradient_accumulation_steps=${gradient_accumulation_steps} --model_checkpoint=${model_checkpoint} --debug=${debug} --n_epochs=${n_epochs} --num_candidates=${num_candidates} --n_workers=${n_workers} --emb_dim=${emb_dim} --add_extra_layer=${add_extra_layer} --iterate_every=${iterate_every} --lr=${lr}

echo "Experiment finished..."
echo "we go to sleep for 2days"
sleep 2d # 1 days!
echo "=============BB guys=============!"
