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
    SEED=$(($SGE_TASK_ID+$RANDOM))	
    #python problem2.py --batch 50000 --samples 1000 --seed $SGE_TASK_ID --fn 0
	#python problem2.py --batch 50000 --samples 1000 --seed $SEED --fn 0
	#python problem2.py --batch 1000 --samples 50000 --seed $SGE_TASK_ID --fn 1
    python problem2.py --batch 100 --samples 50000 --seed $SEED --fn 1
	#python problem2.py --batch 1 --samples 50 --fn 1 --seed $SEED
	#python problem2.py --batch 50000 --samples 1000 --seed $SGE_TASK_ID --fn 2
    fi
done
