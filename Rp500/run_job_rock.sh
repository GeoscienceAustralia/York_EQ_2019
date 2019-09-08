#!/bin/bash
eval "$(conda shell.bash hook)"
for var in rock soil
do
    for rp in Rp500 Rp1000 Rp2500
    do
        ini_file=./$rp/job_$var.ini
        log_file=./$rp/job_$var.log

        if [ -f "$ini_file" ]; then
            oq engine --run $ini_file --exports csv >&  $log_file
            JOB_ID=$(head -n 1 $log_file | grep -Po '(?<=\#).+?')
            echo "$rp, $var, $JOB_ID"
            id_rlz=$(tail -n 4 $log_file | grep -Po '\d+(?= \| Realizations)')
            echo $id_rlz

            oq engine --export-output $id_rlz ./output 
            oq show events > ./$rp/output/eid2rlz_$JOB_ID.csv

            conda activate base
            python ../code/average_gmf.py -i ./output/gmf-data_$JOB_ID.csv
            conda deactivate
        else
            echo "$ini_file does not exist"
        fi
        done
done

