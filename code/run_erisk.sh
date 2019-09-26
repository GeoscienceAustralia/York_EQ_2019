#PBS -m e
#PBS -P y57
#PBS -q normalbw
#PBS -l walltime=24:00:00
#PBS -l ncpus=28
#PBS -l mem=256GB
#PBS -l wd
#PBS -N job_ebrisk
#PBS -l jobfs=1GB
#PBS -l other=hyperthread

module load openquake/3.6
oq-ini.all.sh
ini_file=/home/547/hxr547/Projects/York/input/job_event_risk_multi.ini
log_risk_file=/home/547/hxr547/Projects/York/PSRA/output_37x/job_event_multi.log
oq engine --run $ini_file --exports csv >& $log_risk_file
oq-end.sh
