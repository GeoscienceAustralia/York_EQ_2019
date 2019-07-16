#PBS -m e
#PBS -P y57
#PBS -q normalbw
#PBS -l walltime=00:10:00
#PBS -l ncpus=28
#PBS -l mem=128GB
#PBS -l wd
#PBS -N job_rock
#PBS -l jobfs=10GB
#PBS -l other=hyperthread

module load openquake/3.5
oq-ini.all.sh
oq engine --run job_rock.ini --exports csv,xml >&  job_rock.log && oq engine --export-output 1 ./output
oq-end.sh

