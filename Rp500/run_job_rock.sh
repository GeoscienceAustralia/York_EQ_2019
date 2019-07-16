#PBS -m e
#PBS -P y57
#PBS -q normalbw
#PBS -l walltime=00:10:00
#PBS -l ncpus=56
#PBS -l mem=256GB
#PBS -l wd
#PBS -N job_rock
#PBS -l jobfs=10GB
#PBS -l other=hyperthread

module load openquake/3.5
oq-ini.all.sh
oq engine --run job_rock.ini --exports csv >&  job_rock.log && oq engine --export-output 4 ./output
oq-end.sh

