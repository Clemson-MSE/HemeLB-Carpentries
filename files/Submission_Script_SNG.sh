#!/bin/bash

# Set the name of the job.
#SBATCH -J BifurcationLaunch

#Output and error (also --output, --error):
#SBATCH -o ./%x.%j.out
#SBATCH -e ./%x.%j.err

# Send an email on completion of job
#SBATCH --mail-user=j.mccullough@ucl.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

# walltime, maximum walltime allowed (HH:MM:SS)
#SBATCH --time=0:20:00
#SBATCH --partition=micro

#Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48

#SBATCH --no-requeue
#Setup of execution environment
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --account=pn72qu

# Change to the directory that the job was submitted from
#SBATCH -D ./

#Important
module load slurm_setup

module load intel/19.0.5
module load intel-mpi/2019-intel

module list # Check your modules

# Run the MPI job.
rm -rf $WORK_pn72qu/FocusCOEtests/10c-bif/results5

mpiexec -n $SLURM_NTASKS /dss/dsshome1/00/di39tav2/codes/HemePure/src/build_PP/hemepure -in $WORK_pn72qu/FocusCOEtests/10c-bif/input.xml -out $WORK_pn72qu/FocusCOEtests/10c-bif/results5




