---
title: Compiling HemeLB on your HPC
---

1) Clone or download the source code for HemePure to your HPC
2) Edit the `FullBuildScript.sh` file to pass the correct C++ and MPI compiler shortcuts for your machine
   **N.B.** HemeLB requires C++ and MPI to run, we have successfully tried several different compiler options/versions but cannot guarantee 
   that every combination will work effectively. For open-source options, GNU 7.5.0 and OpenMPI 2.1.1 should provide a good starting point. Python 2.7 is also 
   required for dependency compilation (**TODO??** update to Python3 version?)
3) Run the `FullBuildScript.sh` from the HemePure directory. This will build both the necessary dependencies and the source code for some initial tests.
   The compilation options for improving performance in a later episode (**EDIT ME ADD LINK**). Unless you are changing compiler versions, it is not 
   necessary to rebuild the dependencies each time you wish to test some different code or compilation option.
4) Cross your fingers for compilation to complete. Once finished, there should be the `hemepure` executable in the `src/build` folder.

{% include links.md %}
