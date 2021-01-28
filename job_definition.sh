#!/usr/bin/env bash

project_name='question_generation'        # dont change this
cmd=/project/users/amir/projects/${project_name}/script.sh

experiment='question generation'
model_name_or_path='t5-base'
model_type='t5'
tokenizer_name_or_path='t5_qg_tokenizer'
output_dir='runs/t5-base-e2e-qg-v2-hl-plus-rules'
train_file_path='data/train_data_e2e_qg_v2_highlight_qg_format_t5.pt'
valid_file_path='data/valid_data_e2e_qg_v2_highlight_qg_format_t5.pt'
per_device_train_batch_size=16
per_device_eval_batch_size=16
gradient_accumulation_steps=16
learning_rate=1e-4
num_train_epochs=10
is_debug_mode=False

gpus=4

#let systemMemory=16000




echo "
{
  \"jobDefinition\": {
    \"name\": \"${project_name}\",
    \"description\": \"experiment: ${experiment}\",
    \"dockerImage\": \"gitlab.ilabt.imec.be:4567/ilabt/gpulab-examples/nvidia/sample:nvidia-smi\",
    \"command\": \"${cmd}\",
    \"clusterId\": 6,
    \"resources\": {
      \"gpus\": ${gpus},
      \"systemMemory\":64000,
      \"cpuCores\": ${gpus},
      \"gpuModel\": \"V100\",
      \"minCudaVersion\": \"10\"
    },
    \"jobDataLocations\": [{
      \"mountPoint\": \"/project\"
     }],
    \"portMappings\": [ ],
    \"environment\":
    {
        \"project_name\": \"${project_name}\",

        \"model_name_or_path\": \"${model_name_or_path}\",
        \"model_type\": \"${model_type}\",
        \"tokenizer_name_or_path\": \"${tokenizer_name_or_path}\",
        \"output_dir\": \"${output_dir}\",
        \"train_file_path\": \"${train_file_path}\",
        \"valid_file_path\": \"${valid_file_path}\",
        \"per_device_train_batch_size\": \"${per_device_train_batch_size}\",
        \"per_device_eval_batch_size\": \"${per_device_eval_batch_size}\",
        \"gradient_accumulation_steps\": \"${gradient_accumulation_steps}\",
        \"learning_rate\": \"${learning_rate}\",
        \"num_train_epochs\": \"${num_train_epochs}\",
        \"is_debug_mode\": \"${is_debug_mode}\",
        \"gpus\": \"${gpus}\"

    }
	}
}" | gpulab-cli submit --project cmsearch;

