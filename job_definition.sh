#!/usr/bin/env bash

project_name='question_generation'        # dont change this
cmd=/project/users/amir/projects/${project_name}/script.sh

experiment='question generation'
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
      \"systemMemory\": '64000',
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
        \"gpus\": \"${gpus}\"

    }
	}
}" | gpulab-cli submit --project cmsearch;

