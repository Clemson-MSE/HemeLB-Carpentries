total = 0, //!< Total time
initialDecomposition, //!< Initial seed decomposition
domainDecomposition, //!< Time spent in parmetis domain decomposition
fileRead, //!< Time spent in reading the geometry description file
reRead, //!< Time spend in re-reading the geometry after second decomposition
unzip, //!< Time spend in un-zipping
moves, //!< Time spent moving things around post-parmetis
parmetis, //!< Time spent in Parmetis
latDatInitialise, //!< Time spent initialising the lattice data
lb, //!< Time spent doing the core lattice boltzman simulation
lb_calc, //!< Time spent doing calculations in the core lattice boltzmann simulation
monitoring, //!< Time spent monitoring for stability, compressibility, etc.
mpiSend, //!< Time spent sending MPI data
mpiWait, //!< Time spent waiting for MPI
simulation, //!< Total time for running the simulation