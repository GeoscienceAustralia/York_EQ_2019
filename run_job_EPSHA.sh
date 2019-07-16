#PBS -m e
#PBS -P y57
#PBS -q normalbw
#PBS -l walltime=01:00:00
#PBS -l ncpus=84
#PBS -l mem=768GB
#PBS -l wd
#PBS -N epsha
#PBS -l jobfs=500GB
#PBS -l other=hyperthread

module load openquake/3.5
oq-ini.all.sh
oq engine --run job_EPSHA.ini --exports xml >&  job_EPSHA_eps0.log
oq-end.sh
