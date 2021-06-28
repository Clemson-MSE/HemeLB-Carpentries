#!/bin/bash --login

# PBS job options (name, compute nodes (each node has 24 cores), job time)
# PBS -N is the job name (e.g. Example_MPI_Job)
#PBS -N jobname
# PBS -l select is the number of nodes requested (here 2 nodes with 24 CPUs/node)
#PBS -l select=2:ncpus=24
# PBS -l walltime, maximum walltime allowed (e.g. 15 minutes)
#PBS -l walltime=00:15:00

# Replace [budget code] below with your project code (e.g. t01)
#PBS -A [insert your_projectID_here]

# Make sure any symbolic links are resolved to absolute path
export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)               

# Change to the directory that the job was submitted from
# (remember this should be on the /work filesystem)
cd $PBS_O_WORKDIR

# Set the number of threads to 1
#   This prevents any system libraries from automatically 
#   using threading.
export OMP_NUM_THREADS=1

#Add 'Module load' statements here - check your local system for specific modules/names
module load <Something for local system>
module load <Compiler library used>
module load <MPI library used>

# Launch the parallel job with requested nodes (-n) and cores/node (-N)
# Note that flags may change for different launch tools (e.g. mpirun, srun or mpiexec)
aprun -n 2 -N 24 Path/to/executable/hemepure -in input.xml -out results