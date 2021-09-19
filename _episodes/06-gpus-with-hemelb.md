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

# GPUs - Why do we need GPUs?

A Graphics Processing Unit (GPU) is a type of specialised processor originally designed to accelerate graphics rendering. However, it was gradually 
realised that a GPU can also be used to accelerate other types of calculations as well, involving massive amounts of data, due to the way that it 
is designed to operate. A GPU has nowadays hundreds to thousands processing cores. For example the NVIDIA A100 GPU has 6912 CUDA cores and 432 Tensor
cores. We will not discuss the difference between the 2 now, but what does this mean is that a GPU has much more processing power to complete a given task.  

If you need to perform a task on massive amounts of data, then the same analysis (calculations - set of code) will be executed on/for each one of the 
elements/data that we have. A CPU would have to go through each one of the elements in a serial manner, i.e. perform the analysis on the first element,
once finished move to the next one and so on and so forth, until it manages to process everything. 
A GPU on the other hand, will do this in a parallel way (large scale parallelism), depending on how may cores it has. The same mathematical function
will run over and over again but at a large scale, offering significant speed-up to the calculations.   

A nice demonstration of the above was given by the (MythBusters)[https://www.youtube.com/watch?v=0udMBdo0Rac] at an NVIDIA conference: 

Hence, in scientific computing, with GPUs we can achieve massive acceleration of our calculations. That is why GPUs are becoming commonplace
on high-end HPC machines, with a number of GPUs installed on each node.  

![image](https://user-images.githubusercontent.com/52040752/133001824-ac80d147-8444-4650-9a13-5c0b3ae53f68.png)

The schematic Figure from NVIDIA (documentation)[https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html] shows an example distribution
of chip resources for a CPU versus a GPU.  


## HemeLB and GPUs

A GPU accelerated version of HemeLB has been developed using NVIDIA's CUDA platform. CUDA stands for Compute Unified Device Architecture; it is a
parallel computing platform and application programming interface model created by Nvidia. Hence our GPU HemeLB code is GPU-aware; it can only 
run on NVIDIA's GPUs. 

CUDA does not require developers to have specialised graphics programming knowledge. Developers can use popular programming languages, such as 
C/C++ and Fortran to exploit the GPU resources. The GPU accelerated version of HemeLB was developed using CUDA C++. 

# Introduction to CUDA

## CUDA Programming Basics

The main thing that someone needs to have in mind when it comes to CUDA and GPU programming, is that the compute intensive parts of a code can be 
ported onto the GPU (device) for the calculations to take place for a fraction of the time it would take to complete on a CPU and then get the 
results back to the CPU (host). Hence, the developer needs to implement the GPU CUDA kernels, which are the functions for doing the calculations 
on the GPU, but also arrange: a) the data transfers to and from the GPU, as well as b) the synchronisation points, i.e. when to stop the code moving
past a given point until a certain task on the GPU has been completed.

With the above in mind, a typical sequence of operations for a CUDA C++ program is:
1. Declare and allocate host (CPU) and device (GPU) memory.
2. Initialize host data.
3. Transfer data from the host to the device.
4. Execute one or more CUDA kernels (computations performed on the GPU).
5. Transfer results from the device to the host. 


**GPU CUDA kernel - CUDA function** 

The specifier `__global__` is added in front of the function, which tells the CUDA C++ compiler that this is a function that runs on the GPU and can
be called from CPU code.

A full example of defining a GPU CUDA kernel would be as follows;

~~~
__global__ void GPU_Cuda_Kernel_Name(kernel's_arguments)
~~~
{: .source}

**Launch the GPU kernel**

The GPU CUDA kernel is launched by using a specific syntax, the **triple angle bracket** `<<< >>>`. This will inform the compiler that the kernel
that follows is a GPU kernel and will therefore be executed on the GPU. The information between the triple angle brackets is the execution
configuration, which determines how many threads on the device (GPU) will execute the kernel in parallel. These threads are arranged in thread blocks,
hence the developer should specify how many threads there are per block.   

An example of a launching a GPU CUDA kernel (GPU_Cuda_Kernel_Name) is as follows: 

~~~
// Cuda kernel set-up
int nThreadsPerBlock = 128; //Number of threads per block
dim3 nThreads(nThreadsPerBlock);
 
// Number of BLOCKS needed 
int nBlocks = (number_of_Elements)/nThreadsPerBlock	+ ((number_of_Elements % nThreadsPerBlock > 0) ? 1 : 0);

// Launch the GPU CUDA kernel
GPU_Cuda_Kernel_Name <<< nBlocks, nThreads>>> (Provide_Arguments_here); 
~~~
{: .source}

The first argument in the execution configuration `GPU_Cuda_Kernel_Name **<<< nBlocks, nThreads>>>` specifies the number of thread
blocks (`nBlocks`), while (`nThreads`) specifies the number of threads in a thread block.

Remember that CUDA kernel launches don’t block the calling CPU thread. This means that once the kernel is launched, the control is returned
to the CPU thread and the code will resume. In order to ensure that the GPU kernel has completed its task, a synchronsation barrier might be
needed using `cudaDeviceSynchronize()`.

A discussion on the above as well as simple examples can be found from NVIDIA's website 

https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/

as well as from NVIDIA's CUDA Toolkit Documentation 

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html


## GPU Memory Hierarchy - Memory Allocation in CUDA

CUDA threads can access data from multiple memory spaces. As specified in NVIDIA's CUDA Toolkit Documentation 

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

CUDA threads may access data from multiple memory spaces during their execution (see Figure below). Each thread has private local memory.
Each thread block has shared memory visible to all threads of the block and with the same lifetime as the block. All threads have access to
the same global memory.

There are also two additional read-only memory spaces accessible by all threads: the constant and texture memory spaces. The global, constant,
and texture memory spaces are optimized for different memory usages (see Device Memory Accesses). Texture memory also offers different addressing
modes, as well as data filtering, for some specific data formats (see Texture and Surface Memory).

The global, constant, and texture memory spaces are persistent across kernel launches by the same application. The figure below is from NVIDIA's 
CUDA Toolkit Documentation - GPU Memory hierarchy
![image](https://user-images.githubusercontent.com/52040752/133094845-0b902979-f6a9-48c0-8b93-5be7546f8e48.png)

## CUDA Streams and Concurrency 

A CUDA kernel when launched is placed in the default CUDA stream. A CUDA stream is like an execution path. All operations in the same stream need
to be completed and executed in order, before moving forward.

It is also possible, however, to launch kernels in different CUDA streams and by doing so these kernels may execute out of order with respect to 
one another or concurrently. 
An example of a launching a GPU CUDA kernel (`GPU_Cuda_Kernel_Name`) in a specified CUDA stream (name of stream `CUDA_stream_ID`) is provided here: 

~~~
// Launch the GPU CUDA kernel in stream CUDA_stream_ID
GPU_Cuda_Kernel_Name <<< nBlocks, nThreads, 0, CUDA_stream_ID>>> (Provide_Arguments_here); 
~~~
{: .source}

## Data Transfers in CUDA C/C++

As mentioned above, when performing calculations on the GPU, memory needs to be allocated onto the GPU (`cudaMalloc()`); then data that will be processed needs to be copied from the host to the device (`cudaMemcpyHostToDevice`), perform the calculations (execute the CUDA kernels on the device/GPU) and finally copy the results from the device to the host (`cudaMemcpyDeviceToHost`). 
Data transfers are performed using `cudaMemcpy` function. The syntaxt of `cudaMemcpy` is as follows:

`cudaMemcpy(void *dst, void *src, size_t count, cudaMemcpyKind kind)`

`cudaMemcpyKind` is either `cudaMemcpyHostToDevice` or `cudaMemcpyDeviceToHost`. The 2 possible kinds of CUDA memory copies are:

a. H2D: from the Host (CPU) to the Device (GPU)

`cudaMemcpy(d_A, h_A, size_of_h_A_in_Bytes, cudaMemcpyHostToDevice)`

b. D2H: from the Device (GPU) to the Host (CPU)

`cudaMemcpy(h_A, d_A, size_of_d_A_in_Bytes, cudaMemcpyDeviceToHost)`

These memory copies can be Synchronous (as above) or Asynchronous (`cudaMemcpyAsync`). In the case of asynchronous memory copy, the developer should provide the CUDA stream as a last argument to the `cudaMemcpyAsync` function call. This allows overlapping the memory copies with other operations (kernels or mem.copies) on the GPU running on different CUDA streams.     


## Simple CUDA code example
Here, we provide a simple example of a CUDA code. It contains the main features discussed above: allocate input vectors in host memory and initialise them, allocate memory on the GPU, memory copies (H2D and D2H), defining and launching a GPU CUDA kernel.  

~~~
// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}
            
// Host code
int main()
{
    int N = ...;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    // Initialize input vectors
    ...

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int nblocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<nblocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
            
    // Free host memory
    ...
}
~~~
{: .source}


##  Compile CUDA code
CUDA code (typically in a file with extension `.cu`) can be compiled using the `nvcc` compiler. 
For example
~~~
nvcc CUDA_code.cu -o CUDA_code
~~~

## Profiling CUDA code
Profiling the CUDA code can be done using tools provided by NVIDIA.    
**NVIDIA Nsight Systems** for GPU and CPU sampling and tracing and **NVIDIA Nsight Compute** for GPU kernel profiling. 

A more detailed description on the above tools can be provided from NVIDIA's CUDA Toolkit Documentation 

https://docs.nvidia.com/cuda/profiler-users-guide/index.html


> ## A note on GPU Profiling
> 
> It should be noted that some HPC systems do not readily accommodate GPU profiling tools such as NVIDIA Nsight Systems, owing to that fact that root
> directory access may be required. You should check with your system administrator of the different options available to you on your system for 
> profiling GPU code.
{: .callout}


##  Running HemeLB on HPC machines with NVIDIA's GPUs
Job submission scripts (JUWELS Booster) 

https://github.com/HemeLB-dev/HemeLB-Carpentries/blob/gh-pages/files/Submission_Script_Juwels_Booster.sh

and Summit (OLCF)

https://github.com/HemeLB-dev/HemeLB-Carpentries/blob/gh-pages/files/Submission_Script_OLCF.lsf

When submitting a job script on an HPC machine with nodes containing NVIDIA's GPUs, the user should specify the number of GPUs to be used on each node. The format of the submission script depends on the HPC system.  

