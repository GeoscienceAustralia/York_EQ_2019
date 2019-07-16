#PBS -m e
#PBS -P y57
#PBS -q normalbw
#PBS -l walltime=00:10:00
<<<<<<< HEAD
#PBS -l ncpus=56
#PBS -l mem=256GB
=======
#PBS -l ncpus=28
#PBS -l mem=128GB
>>>>>>> 2d9fd18c083841dd10dd1a666de760618446d5ff
#PBS -l wd
#PBS -N job_rock
#PBS -l jobfs=10GB
#PBS -l other=hyperthread

module load openquake/3.5
oq-ini.all.sh
<<<<<<< HEAD
oq engine --run job_rock.ini --exports csv >&  job_rock.log && oq engine --export-output 4 ./output
=======
oq engine --run job_rock.ini --exports csv,xml >&  job_rock.log && oq engine --export-output 1 ./output
>>>>>>> 2d9fd18c083841dd10dd1a666de760618446d5ff
oq-end.sh

