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
oq engine --run ./Rp500/job_rock.ini --exports csv,xml >&  ./Rp500/job_rock.log && oq engine --export-output 4 ./Rp500/output
oq engine --run ./Rp1000/job_rock.ini --exports csv,xml >&  ./Rp1000/job_rock.log && oq engine --export-output 8 ./Rp1000/output
oq engine --run ./Rp2500/job_rock.ini --exports csv,xml >&  ./Rp2500/job_rock.log && oq engine --export-output 12 ./Rp2500/output
oq-end.sh

