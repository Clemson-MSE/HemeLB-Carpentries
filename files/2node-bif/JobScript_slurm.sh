#!/bin/bash
# Job Name and Files
#SBATCH -J jobname

#Output and error:
#SBATCH -o ./%x.%j.out
#SBATCH -e ./%x.%j.err
#Initial working directory:
#SBATCH -D ./

# Wall clock limit:
#SBATCH --time=00:15:00

#Setup of execution environment <Check/Set as appropriate for your local system>
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --account=insert your_projectID_here
#SBATCH --partition=<CheckLocalSystem>

#Number of nodes and MPI tasks per node:
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24

#Add 'Module load' statements here - check your local system for specific modules/names
module load <Something for local system>
module load <Compiler library used>
module load <MPI library used>

#Run HemeLB (other systems may use srun or mpirun in place of mpiexec):
rm -rf results #delete old results file
mpiexec -n $SLURM_NTASKS Path/to/executable/hemepure -in input.xml -out results

