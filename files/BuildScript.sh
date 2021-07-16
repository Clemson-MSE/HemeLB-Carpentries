#!/bin/bash
## Compilation/build script for PLEPP-HEMELB
## Run from found location

## MODULE loads
##GCC compilers

export FC=mpif90 && export CC=mpicc && export CXX=mpicxx

## HEMELB build
# 1) Dependencies
BuildDep(){
cd dep
rm -rf build
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} ..
make -j  && echo "Done HemeLB Dependencies"
cd ../..
}

# 2) Source code
BuildSource(){
cd src
rm -rf build
mkdir build
cd build

cmake ../  \ 
  -DCMAKE_C_COMPILER=${CC} \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DHEMELB_USE_SSE3=OFF\
  -DHEMELB_KERNEL="LBGK"\
  -DHEMELB_WALL_BOUNDARY="SIMPLEBOUNCEBACK"\
  -DHEMELB_INLET_BOUNDARY="NASHZEROTHORDERPRESSUREIOLET"\
  -DHEMELB_WALL_INLET_BOUNDARY="NASHZEROTHORDERPRESSURESBB"\
  -DHEMELB_OUTLET_BOUNDARY="NASHZEROTHORDERPRESSUREIOLET"\
  -DHEMELB_WALL_OUTLET_BOUNDARY="NASHZEROTHORDERPRESSURESBB"\
  -DHEMELB_USE_VELOCITY_WEIGHTS_FILE=OFF


make -j && echo "Done HemeLB Source"

cd ../..
}

#BuildDep
BuildSource

echo "Done build all"


