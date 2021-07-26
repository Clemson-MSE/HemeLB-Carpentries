---
title: "GPUs with HemeLB"
teaching: 15
exercises: 15
questions:
- "How do I use the HemeLB library on a GPU?"
objectives:
- "What is the performance like?"
keypoints:
- "Knowing the capabilities of your host, device and if you can use a CUDA-aware MPI
  runtime is required before starting a GPU run"
---


**GPUs - General Introduction - Introduction to CUDA**


**CUDA Programming Basics**
A typical sequence of operations for a CUDA C++ program is:
1. Declare and allocate host (CPU) and device (GPU) memory.
2. Initialize host data.
3. Transfer data from the host to the device.
4. Execute one or more kernels (computations performed on the GPU).
5. Transfer results from the device to the host.


GPU kernel - CUDA function 
Add the specifier __global__ in front of the function, which tells the CUDA C++ compiler that this is a function that runs on the GPU and can be called from CPU code.

Launch the GPU kernel  (triple angle bracket syntax <<< >>>)
Remember that CUDA kernel launches don’t block the calling CPU thread. 
Wait the GPU kernel to complete its task - Use of cudaDeviceSynchronize()

CUDA files (extension .cu)
Compile CUDA code (nvcc - CUDA C++ compiler), e.g. nvcc cuda_example.cu -o cuda_example



**GPU Memory Hierarchy - Memory Allocation in CUDA** 
GPU Global memory, constant memory

At the end of this section mention about Unified Memory in CUDA (provides  a single memory space accessible by all GPUs and CPUs in the system).

**CUDA Streams and Concurrency**


**Data Transfers in CUDA C/C++**
CUDA memory copies:
a. D2H: from the Device (GPU) to the Host (CPU) 
b. H2D: from the Host (CPU) to the Device (GPU)

Can be Synchronous or Asynchronous... Explain... 



**Profiling CUDA code**
NVIDIA Nsight Systems




