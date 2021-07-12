#!/bin/bash -x

# Job options (Account, compute nodes (each node has 4 GPUs), number of MPI tasks (assume 1:1 ratio CPU_cores(or MPI_tasks):GPU_cores), 
#               Output and error files, job time)

#SBATCH --account=prcoe06
#SBATCH --nodes=16
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=4
#SBATCH --output=hemelb_gpu-out.%j
#SBATCH --error=hemelb_gpu-err.%j
#SBATCH --time=01:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4

#Add 'Module load' statements here - check your local system for specific modules/names
module load <Something for local system>
module load <Compiler library used>
module load <MPI library used>

# For example:
# Compiler:     module load GCC/9.3.0
# MPI library:  module load ParaStationMPI/5.4.7-1
#                 OR module load OpenMPI
# For GPUs:     module load CUDA

# Add the details for the executable to run and the input file
EXEC=<Provide executable name>                # e.g. hemepure_gpu
EXEC_PATH=<Provide Path to executable>        # /p/project/prcoe06/code_GPU/GNU_ParastationMPI/src_v1_27/build
INPUT_FILE=<Provide name of the input file>  # input.xml

rm -r results;

srun $EXEC_PATH/$EXEC -in $INPUT_FILE -out results
