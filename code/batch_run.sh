#!/bin/bash
eval "$(conda shell.bash hook)"

PROJ_PATH=/home/hyeuk/Projects/York
OUTPUT='output_37x'

#find $PROJ_PATH -name $OUTPUT -exec rm -rf {} \;

EVENT='Rp500 Rp1000 Rp2500'
YEAR='10 20 30'
LEVEL='1 2 3 4'
SCHEME='1 2'

run_risk() {

    local event=$1
    local scheme=$2
    local year=$3
    local job_id=$4

    for level in $LEVEL
    do    
        ini_file=$PROJ_PATH/$event/job_risk.ini
        log_risk_file=$PROJ_PATH/$event/$OUTPUT/job_risk_S"$scheme"_Y"$year"_L"$level".log

        cp $PROJ_PATH/input/job_risk.ini $ini_file

        sed -i 's/<EVENT>/'"$event"'/g' $ini_file
        sed -i 's/<SCHEME>/'"$scheme"'/g' $ini_file
        sed -i 's/<YEAR>/'"$year"'/g' $ini_file
        sed -i 's/<LEVEL>/'"$level"'/g' $ini_file
        sed -i 's/<GMID>/'"$job_id"'/g' $ini_file
        sed -i 's/<OUTPUT>/'"$OUTPUT"'/g' $ini_file

        echo "running oq risk: $event, $scheme, $year, $level, $job_id"
        oq engine --run $ini_file --exports csv >& $log_risk_file
    done
}

run_dmg() {

    local event=$1
    local scheme=$2
    local year=$3
    local job_id=$4

    ini_file=$PROJ_PATH/$event/job_damage.ini
    log_dmg_file=$PROJ_PATH/$event/$OUTPUT/job_dmg_S"$scheme"_Y"$year".log

    cp $PROJ_PATH/input/job_damage.ini $ini_file

    sed -i 's/<EVENT>/'"$event"'/g' $ini_file
    sed -i 's/<SCHEME>/'"$scheme"'/g' $ini_file
    sed -i 's/<YEAR>/'"$year"'/g' $ini_file
    sed -i 's/<GMID>/'"$job_id"'/g' $ini_file
    sed -i 's/<OUTPUT>/'"$OUTPUT"'/g' $ini_file

    echo "running oq dmg : $event, $scheme, $year, $job_id"
    oq engine --run $ini_file --exports csv >& $log_dmg_file
}

run_gmf() {

    local event=$1
    local var=$2

    ini_file=$PROJ_PATH/$event/job_$var.ini
    log_file=$PROJ_PATH/$event/$OUTPUT/job_$var.log

    # create ini file
    cp $PROJ_PATH/input/job_$var.ini $ini_file
    sed -i 's/<EVENT>/'"$event"'/g' $ini_file
    sed -i 's/<OUTPUT>/'"$OUTPUT"'/g' $ini_file

    # run oq engine
    oq engine --run $ini_file --exports csv >&  $log_file &&
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

    return $JOB_ID
}

run_prob_risk() {
    
    local var=$1
    local scheme=$2
    local year=$3

    ini_file=$PROJ_PATH/PSRA/job_"$var"_risk.ini
    log_risk_file=$PROJ_PATH/PSRA/$OUTPUT/job_"$var"_risk_S"$scheme"_Y"$year".log

    cp $PROJ_PATH/input/job_"$var"_risk.ini $ini_file

    sed -i 's/<SCHEME>/'"$scheme"'/g' $ini_file
    sed -i 's/<YEAR>/'"$year"'/g' $ini_file
    sed -i 's/<OUTPUT>/'"$OUTPUT"'/g' $ini_file

    echo "running oq prob risk: $var, $scheme, $year"
    oq engine --run $ini_file --exports csv >& $log_risk_file
}

# main loop
for event in $EVENT
do
    mkdir -p $PROJ_PATH/$event/$OUTPUT;

    # gmf
    for var in rock soil
    do
        run_gmf $event $var
    done

    job_id=$?

    # as-it-is
    run_risk $event 0 0 $job_id
    run_dmg $event 0 0 $job_id

    # retrofit
    for scheme in $SCHEME
    do
        for year in $YEAR
        do
            run_risk $event R"$scheme" $year $job_id
            run_dmg $event R"$scheme" $year $job_id
        done
    done
done

# prob risk
#array1=("0" "R1" "R2")
#array2=("0" "30" "30")
#length=${#array1[@]}
##for var in classical event
#mkdir -p $PROJ_PATH/PSRA/$OUTPUT;
#for var in event
#do 
#    for ((i=0;i<$length;i++)); do
#        run_prob_risk $var ${array1[$i]} ${array2[$i]}
#    done
#done

