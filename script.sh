#!/usr/bin/env bash
echo "=============Hello guys=============!"

PROJECT_NAME=${project_name}
PROJECT_PATHS=/project/users/amir/projects
#PROJECT_SCRATCH_PATHS=/project_scratch/users/amir/projects

export PATH=/project/users/amir/miniconda/conda/bin:$PATH
export PATH=$PROJECT_PATHS/$PROJECT_NAME:$PATH

echo "=============python path set=============!"
#python --version # 3.7.3

cd $PROJECT_PATHS/$PROJECT_NAME || exit

echo "=============install requirements=============!"
apt-get update && apt-get install -y rsync && apt-get install -y libffi6



python3 -m pip install -r requirements.txt || exit


echo "=============installation done!!!=============!"


echo "=============preprocess=============!"

python3 preprocess.py --dataset_path=${dataset_name} --train_batch_size=${train_batch_size} --valid_batch_size=${valid_batch_size} --gradient_accumulation_steps=${gradient_accumulation_steps} --model_checkpoint=${model_checkpoint} --debug=${debug} --n_epochs=${n_epochs} --num_candidates=${num_candidates} --n_workers=${n_workers} --emb_dim=${emb_dim} --add_extra_layer=${add_extra_layer} --iterate_every=${iterate_every} || exit

echo "=============training=============!"
python3 -m torch.distributed.launch --nproc_per_node=${gpus} train.py --dataset_path=${dataset_name} --train_batch_size=${train_batch_size} --valid_batch_size=${valid_batch_size} --gradient_accumulation_steps=${gradient_accumulation_steps} --model_checkpoint=${model_checkpoint} --debug=${debug} --n_epochs=${n_epochs} --num_candidates=${num_candidates} --n_workers=${n_workers} --emb_dim=${emb_dim} --add_extra_layer=${add_extra_layer} --iterate_every=${iterate_every} --lr=${lr}

echo "Experiment finished..."
echo "we go to sleep for 2days"
sleep 10d # 1 days!
echo "=============BB guys=============!"
