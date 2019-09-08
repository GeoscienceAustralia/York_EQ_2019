#PBS -m e
#PBS -P y57
#PBS -q normalbw
#PBS -l walltime=01:00:00
#PBS -l ncpus=84
#PBS -l mem=768GB
#PBS -l wd
#PBS -N psha
#PBS -l jobfs=500GB
#PBS -l other=hyperthread

module load openquake/3.5
oq-ini.all.sh
oq engine --run job_PSHA.ini --exports csv >&  job_PSHA_v35_eps0.log
oq-end.sh
