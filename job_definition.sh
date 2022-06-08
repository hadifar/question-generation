#!/bin/bash

project_name='question_generation'        # dont change this

cmd=/project/users/amir/projects/${project_name}/script.sh

experiment='question generation'
gpus=1

#model_name_or_path='t5-base'
#
#model_type='t5'
#
##tokenizer_name_or_path='t5_qg_tokenizer'
#
#output_dir='runs/squad_qg'
#
#train_file_path='raw_data/dev-2.0.json'
#valid_file_path='raw_data/train-2.0.json'
#
#per_device_train_batch_size=8
#per_device_eval_batch_size=8
#learning_rate=6.25e-5
#num_train_epochs=10
#
#is_debug_mode=-1
#
#
##let systemMemory=16000


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
      \"systemMemory\":16000,
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
        \"project_name\": \"${project_name}\"
    }
	}
}" | gpulab-cli submit --project cmsearch;

