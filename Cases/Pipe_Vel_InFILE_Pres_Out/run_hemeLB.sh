#!/bin/bash

PATH_EXE={Provide/the/path/to/the_executable}
EXEC_FILE={Executable_filename_e.g.hemepure}

INPUT_FILE=input.xml

rm -rf results

mpirun -np 4 $PATH_EXE/$EXEC_FILE -in $INPUT_FILE -out results
