#PBS -m e
#PBS -P y57
#PBS -q normalbw
#PBS -l walltime=24:00:00
#PBS -l ncpus=28
#PBS -l mem=256GB
#PBS -l wd
#PBS -N job_crisk
#PBS -l jobfs=1GB
#PBS -l other=hyperthread

module load openquake/3.6
oq-ini.all.sh
source /home/547/hxr547/Projects/York/code/batch_run.sh
oq-end.sh
