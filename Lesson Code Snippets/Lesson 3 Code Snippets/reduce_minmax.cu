#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <ctime>
#include <cuda_runtime.h>

__global__ void shmem_reduce_kernel(float * d_out, const float * const d_in, bool is_max)
{
  // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
  extern __shared__ float sdata[];

  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid  = threadIdx.x;

  // load shared mem from global mem
  sdata[tid] = d_in[myId];
  __syncthreads();            // make sure entire block is loaded!

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
      if (tid < s)
      {
          if(is_max)
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
          else
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
      }
      __syncthreads();        // make sure all adds at one stage are done!
  }

  // only thread 0 writes result for this block back to global mem
  if (tid == 0)
  {
      d_out[blockIdx.x] = sdata[0];
  }
}

void reduce(float *min_logLum, float *max_logLum, const float* const d_logLuminance, int length)
{
  // use reduce
  const int m = 1 << 10;
  int blocks = ceil((float)length / m);
  float *d_intermediate; // should not modify d_in
  cudaMalloc(&d_intermediate, sizeof(float)* blocks); // store max and min

  shmem_reduce_kernel<<<blocks, m, m * sizeof(float)>>>(d_intermediate, d_logLuminance, true);
  shmem_reduce_kernel<<<1, blocks, blocks * sizeof(float)>>>(max_logLum, d_intermediate, true);

  shmem_reduce_kernel<<<blocks, m, m * sizeof(float)>>>(d_intermediate, d_logLuminance, false);
  shmem_reduce_kernel<<<1, blocks, blocks * sizeof(float)>>>(min_logLum, d_intermediate, false);

  cudaFree(d_intermediate);
}

int main(int argc, char **argv)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int)devProps.totalGlobalMem, 
               (int)devProps.major, (int)devProps.minor, 
               (int)devProps.clockRate);
    }

    const int ARRAY_SIZE = 1 << 16;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // generate the input array on the host
    float h_in[ARRAY_SIZE];
    float sum = 0.0f;
    srand((unsigned)time(0));
    for(int i = 0; i < ARRAY_SIZE; i++) {
        // generate random float in [-1.0f, 1.0f]
        h_in[i] = -1.0f + (float)rand()/((float)RAND_MAX/2.0f);
        sum += h_in[i];
    }

    // declare GPU memory pointers
    float *d_in;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // launch the kernel
    printf("Running reduce\n");
    float *d_min, *d_max;
    cudaMalloc((void **) &d_min, sizeof(float));
    cudaMalloc((void **) &d_max, sizeof(float));
    reduce(d_min, d_max, d_in, ARRAY_SIZE);

    // copy back the sum from GPU
    float h_min, h_max;
    cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Max_GPU: %f  Min_GPU: %f\n", h_max, h_min);
    h_max = h_in[0]; h_min = h_in[0];
    for (size_t i = 1; i < ARRAY_SIZE; ++i) {
        h_max = std::max(h_in[i], h_max);
        h_min = std::min(h_in[i], h_min);
    }
    printf("Max_CPU: %f  Min_CPU: %f\n", h_max, h_min);

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_min);
    cudaFree(d_max);
    return 0;
}
