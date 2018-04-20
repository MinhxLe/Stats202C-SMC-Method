#!/bin/sh
#$ -cwd
#$ -e nn_saw.log
#$ -o nn_saw.log
#$ -j y
#$ -l h_data=1G,h_rt=23:00:00,highp
#$ -t 1-200:1

#SGE_TASK_ID=1

source /u/local/Modules/default/init/modules.sh
module load anaconda/python3-4.2

for i in {1..200}
do
    COUNTER=$((COUNTER+1))
    if [[ $COUNTER -eq $SGE_TASK_ID ]]
    then
	python problem2.py --batch 1000 --samples 50000 --seed $SGE_TASK_ID --fn 0 
fi
