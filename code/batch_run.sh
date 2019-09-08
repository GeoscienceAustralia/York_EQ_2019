#!/bin/bash
eval "$(conda shell.bash hook)"

PROJ_PATH=/home/hyeuk/Projects/York
OUTPUT='output'

EVENT='Rp500 Rp1000 Rp2500'
YEAR='0 10 20 30'
LEVEL='1 2 3 4'

for event in $EVENT
do
    mkdir -p $PROJ_PATH/$event/$OUTPUT;

    # gmf
    for var in rock soil
    do
        ini_file=$PROJ_PATH/$event/job_$var.ini
        log_file=$PROJ_PATH/$event/$OUTPUT/job_$var.log

        # create ini file
        cp $PROJ_PATH/input/job_$var.ini $ini_file
        sed -i 's/<EVENT>/'"$event"'/g' $ini_file
        sed -i 's/<OUTPUT>/'"$OUTPUT"'/g' $ini_file

        # run oq engine
        oq engine --run $ini_file --exports csv >&  $log_file
        JOB_ID=$(head -n 1 $log_file | grep -Po '(?<=\#)\d+')
        echo "$event, $var, $JOB_ID"
        id_rlz=$(tail -n 4 $log_file | grep -Po '\d+(?= \| Realizations)')
        echo $id_rlz

        oq engine --export-output $id_rlz $PROJ_PATH/$event/$OUTPUT
        oq show events > $PROJ_PATH/$event/$OUTPUT/eid2rlz_$JOB_ID.csv

        # compute average gmf
        conda activate base
        python $PROJ_PATH/code/average_gmf.py -i $PROJ_PATH/$event/$OUTPUT/gmf-data_$JOB_ID.csv
        conda deactivate
    done

    for year in $YEAR
    do
        # risk
        for level in $LEVEL
        do
            log_risk_file=$PROJ_PATH/$event/$OUTPUT/job_risk_Y"$year"_L"$level".log
            ini_file=$PROJ_PATH/$event/job_risk.ini

            cp $PROJ_PATH/input/job_risk.ini $ini_file

            sed -i 's/<EVENT>/'"$event"'/g' $ini_file
            sed -i 's/<YEAR>/'"$year"'/g' $ini_file
            sed -i 's/<LEVEL>/'"$level"'/g' $ini_file
            sed -i 's/<GMID>/'"$JOB_ID"'/g' $ini_file
            sed -i 's/<OUTPUT>/'"$OUTPUT"'/g' $ini_file

            echo "running oq using $ini_file : $year, $level"
            oq engine --run $ini_file --exports csv >& $log_risk_file
        done

        # damage
        log_dmg_file=$PROJ_PATH/$event/$OUTPUT/job_dmg_Y"$year".log
        ini_file=$PROJ_PATH/$event/job_damage.ini

        cp $PROJ_PATH/input/job_damage.ini $ini_file

        sed -i 's/<EVENT>/'"$event"'/g' $ini_file
        sed -i 's/<YEAR>/'"$year"'/g' $ini_file
        sed -i 's/<GMID>/'"$JOB_ID"'/g' $ini_file
        sed -i 's/<OUTPUT>/'"$OUTPUT"'/g' $ini_file

        echo "running oq using $ini_file : $year"
        oq engine --run $ini_file --exports csv >& $log_dmg_file
    done

done

