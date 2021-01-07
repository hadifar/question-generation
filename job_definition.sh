#!/usr/bin/env bash

project_name='question_generation'        # dont change this
cmd=/project/users/amir/projects/${project_name}/script.sh

# xlm-mlm-17-1280
# xlm-mlm-tlm-xnli15-1024
# xlm-roberta-base

debug=0
add_extra_layer=1
name="${model_checkpoint}_extra_layer_${add_extra_layer}"
experiment='xlm-roberta'
gpus=3
let systemMemory=16000*${gpus}

echo "
{
  \"jobDefinition\": {
    \"name\": \"${name}\",
    \"description\": \"experiment: ${experiment}\",
    \"dockerImage\": \"gitlab.ilabt.imec.be:4567/ilabt/gpulab-examples/nvidia/sample:nvidia-smi\",
    \"command\": \"${cmd}\",
    \"clusterId\": 6,
    \"resources\": {
      \"gpus\": ${gpus},
      \"systemMemory\": ${systemMemory},
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
        \"model_checkpoint\": \"${model_checkpoint}\",
        \"dataset_name\": \"${dataset_name}\",
        \"n_epochs\": \"${n_epochs}\",
        \"gradient_accumulation_steps\": \"${gradient_accumulation_steps}\",
        \"train_batch_size\": \"${train_batch_size}\",
        \"valid_batch_size\": \"${valid_batch_size}\",
        \"num_candidates\": \"${num_candidates}\",
        \"emb_dim\": \"${emb_dim}\",
        \"iterate_every\": \"${iterate_every}\",
        \"lr\": \"${lr}\",
        \"n_workers\": \"${n_workers}\",
        \"add_extra_layer\": \"${add_extra_layer}\",
        \"debug\": \"${debug}\",
        \"gpus\": \"${gpus}\"

    }
	}
}" | gpulab-cli submit --project cmsearch;

