#!/bin/bash

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

echo "=============RUN QG=============!"

bash run.sh

echo "Experiment finished..."
echo "we go to sleep for 2days"
sleep 1d # 1 days!
echo "=============BB guys=============!"
