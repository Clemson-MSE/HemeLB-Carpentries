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

**GPUs - Why do we need GPUs?**

A Graphics Processing Unit (GPU) is a type of specialised processor originally designed to accelerate graphics rendering. However, it was gradually realised that a GPU can also be used to accelerate other types of calculations as well, involving massive amounts of data, due to the way that it is designed to operate. 
A GPU has nowadays hundreds to thousands processing cores. For example the NVIDIA A100 GPU has 6912 CUDA cores and 432 Tensor cores. We will not discuss the difference between the 2 now, but what does this mean is that a GPU has much more processing power to complete a given task.  

If you need to perform a task on massive amounts of data, then the same analysis (calculations - set of code) will be executed on/for each one of the elements/data that we have. A CPU would have to go through each one of the elements in a serial manner, i.e. perform the analysis on the first element, once finished move to the next one and so on and so forth, until it manages to process everything. 
A GPU on the other hand, will do this in a parallel way (large scale parallelism), depending on how may cores it has. The same mathematical function will run over and over again but at a large scale, offering significant speed-up to the calculations.   

A nice demonstration of the above was given by the MythBusters: 
"MythBusters feat. NVidia: GPU VS CPU - Mona Lisa PaintBalled" https://www.youtube.com/watch?v=0udMBdo0Rac

Hence, in scientific computing, with GPUs we can achieve massive acceleration of our calculations. That is why GPUs are becoming commonplace on high-end HPC machines, with a number of GPUs installed on each node.  

![image](https://user-images.githubusercontent.com/52040752/133001824-ac80d147-8444-4650-9a13-5c0b3ae53f68.png)
Figure from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
The schematic Figure shows an example distribution of chip resources for a CPU versus a GPU.  


**HemeLB and GPUs**

A GPU accelerated version of HemeLB has been developed using NVIDIA's CUDA platform. CUDA stands for Compute Unified Device Architecture; it is a parallel computing platform and application programming interface model created by Nvidia. Hence our GPU HemeLB code is GPU-aware; it can only run on NVIDIA's GPUs. 

CUDA does not require developers to have specialised graphics programming knowledge. Developers can use popular programming languages, such as C/C++ and Fortran to exploit the GPU resources. 
The GPU accelerated version of HemeLB was developed using CUDA C++. 


**GPUs - General Introduction - Introduction to CUDA**


**CUDA Programming Basics**

The main thing that someone needs to have in mind when it comes to CUDA and GPU programming, is that the compute intesive parts of a code can be ported onto the GPU (device) for the calculations to take place for a fraction of the time it would take to complete on a CPU and then get the results back to the CPU (host). Hence, the developer needs to implement the GPU CUDA kernels, which are the functions for doing the calculations on the GPU, but also arrange: a) the data transfers to and from the GPU, as well as b) the synchronisation points, i.e. when to stop the code moving past a given point until a certain task on the GPU has been completed.

With the above in mind, a typical sequence of operations for a CUDA C++ program is:
1. Declare and allocate host (CPU) and device (GPU) memory.
2. Initialize host data.
3. Transfer data from the host to the device.
4. Execute one or more CUDA kernels (computations performed on the GPU).
5. Transfer results from the device to the host. 


**GPU CUDA kernel - CUDA function** 

The specifier __global__ is added in front of the function, which tells the CUDA C++ compiler that this is a function that runs on the GPU and can be called from CPU code.


**Launch the GPU kernel**

The GPU CUDA kernel is launhed by using a specific syntax (**triple angle bracket syntax <<< >>>**). This will inform the compiler that the kernel that follows is a GPU kernel and will therefore be executed on the GPU. 
Remember that CUDA kernel launches don’t block the calling CPU thread. This means that once the kernel is launched, the control is returned to the CPU thread and the code will resume. In order to ensure that the GPU kernel has completed its task, a synchronsation barrier should be used - **Use of cudaDeviceSynchronize()**.  



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


CUDA files (extension .cu)
Compile CUDA code (nvcc - CUDA C++ compiler), e.g. nvcc cuda_example.cu -o cuda_example



