#include <stdio.h>

// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i < N)
       C[i] = A[i] + B[i];
}

// Initialise the input vectors
void initialise_input_vect(float* A, float* B, int N)
{
  for(int i=0; i<N; i++){
    A[i]=i;
    B[i]=2*i;
  }
}


// Host code
int main()
{
   int N = 1000;   // Number of elements to process
   bool print_results = 0; // Boolean variable for printing the results
   size_t size = N * sizeof(float);
//==========================================================================
// Get the GPUs properties:
//    Device name, Compute Capability, Global Memory (GB) etc
   int nDevices;
   cudaGetDeviceCount(&nDevices);
   for (int i = 0; i < nDevices; i++) {
       cudaDeviceProp prop;
       cudaGetDeviceProperties(&prop, i);
       printf("Device Number: %d\n", i);
       printf("  Device name:        %s\n", prop.name);
       printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
       printf("  Total Global Mem:   %.1fGB\n\n", ((double)prop.totalGlobalMem/1073741824.0));
       printf("  Memory Clock Rate (KHz): %d\n",
              prop.memoryClockRate);
       printf("  Memory Bus Width (bits): %d\n",
              prop.memoryBusWidth);
       printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
              2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

       printf("  Max Number of Threads per Block:  %d\n", prop.maxThreadsPerBlock);
       printf("  Max Number of Blocks allowed in x-dir:  %d\n", prop.maxGridSize[0]);
       printf("  Max Number of Blocks allowed in y-dir:  %d\n", prop.maxGridSize[1]);
       printf("  Max Number of Blocks allowed in z-dir:  %d\n", prop.maxGridSize[2]);
       printf("  Warp Size:  %d\n",  prop.warpSize);
       printf("===============================================\n\n");
    }
//==========================================================================
// Allocate input vectors h_A and h_B in host memory
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    // Initialize input vectors
    initialise_input_vect(h_A, h_B, N);

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
    int nThreadsPerBlock = 256;
    int nblocks = (N / nThreadsPerBlock) + ((N % nThreadsPerBlock > 0) ? 1 : 0);
    VecAdd<<<nblocks, nThreadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the results
    if(print_results) for (int i=0; i<N; i++) printf("h_C[%d] = %2.2f \n", i, h_C[i] );

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A, h_B, h_C;
}
