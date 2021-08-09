#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=BifurcationLaunch
#SBATCH --time=0:20:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1

# Remove first # to send an email on completion of job
##SBATCH --mail-user=emailaddress@institute.ac.uk
##SBATCH --mail-type=BEGIN,END,FAIL

# Replace [budget code] below with your budget code (e.g. t01)
#SBATCH --account=e283
#SBATCH --partition=standard
#SBATCH --qos=standard

module load epcc-job-env
module load PrgEnv-gnu

export OMP_NUM_THREADS=1

rm -rf results

srun --unbuffered --cpu-bind=cores /work/e283/e283/jonmcc2/Cases_Single/PP_HemePure -in input.xml -out results

