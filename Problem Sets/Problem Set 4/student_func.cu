//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <thrust/sort.h>

/* Red Eye Removal
===============

For this assignment we are implementing red eye removal.  This is
accomplished by first creating a score for every pixel that tells us how
likely it is to be a red eye pixel.  We have already done this for you - you
are receiving the scores and need to sort them in ascending order so that we
know which pixels to alter to remove the red eye.

Note: ascending order == smallest to largest

Each score is associated with a position, when you sort the scores, you must
also move the positions accordingly.

Implementing Parallel Radix Sort with CUDA
==========================================

The basic idea is to construct a histogram on each pass of how many of each
"digit" there are.   Then we scan this histogram so that we know where to put
the output of each digit.  For example, the first 1 must come after all the
0s so we have to know how many 0s there are to be able to start moving 1s
into the correct position.

1) Histogram of the number of occurrences of each digit
2) Exclusive Prefix Sum of Histogram
3) Determine relative offset of each digit
For example [0 0 1 1 0 0 1]
->  [0 1 4 5 2 3 6]
4) Combine the results of steps 2 & 3 to determine the final
output location for each element and move it there

LSB Radix sort is an out-of-place sort and you will need to ping-pong values
between the input and output buffers we have provided.  Make sure the final
sorted results end up in the output buffer!  Hint: You may need to do a copy
at the end.

*/

//#define USE_THRUST

__global__ void print_kernel(unsigned int *d_out)
{
  printf("%d ", d_out[threadIdx.x]);
}


__global__ void histo_kernel(unsigned int * d_out, unsigned int* const d_in,
  unsigned int shift, const unsigned int numElems)
{
  unsigned int mask = 1 << shift;
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  if (myId >= numElems)  return;
  int bin = (d_in[myId] & mask) >> shift;
  atomicAdd(&d_out[bin], 1);
}

// Blelloch Scan - described in lecture
__global__ void sumscan_kernel(unsigned int * d_in, const size_t numBins, const unsigned int numElems)
{
  int myId = threadIdx.x;
  if (myId >= numElems)  return;
  extern __shared__ float sdata[];
  sdata[myId] = d_in[myId];
  __syncthreads();            // make sure entire block is loaded!

  for (int d = 1; d < numBins; d *= 2) {
    if (myId >= d) {
      sdata[myId] += sdata[myId - d];
    }
    __syncthreads();
  }
  if (myId == 0)  d_in[0] = 0;
  else  d_in[myId] = sdata[myId - 1]; //inclusive->exclusive
}

__global__ void makescan_kernel(unsigned int * d_in, unsigned int *d_scan,
  unsigned int shift, const unsigned int numElems)
{
  unsigned int mask = 1 << shift;
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  if (myId >= numElems)  return;
  d_scan[myId] = ((d_in[myId] & mask) >> shift) ? 0 : 1;
}

__global__ void move_kernel(unsigned int* const d_inputVals,
  unsigned int* const d_inputPos,
  unsigned int* const d_outputVals,
  unsigned int* const d_outputPos,
  const unsigned int numElems,
  unsigned int* const d_histogram,
  unsigned int* const d_scaned,
  unsigned int shift)
{
  unsigned int mask = 1 << shift;
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  if (myId >= numElems)  return;
  // Important! 
  // Algorithm described in 7.4 of http://wykvictor.github.io/2016/04/03/Cuda-2.html 
  int des_id = 0;
  if ((d_inputVals[myId] & mask) >> shift) {
    des_id = myId + d_histogram[1] - d_scaned[myId];
  } else {
    des_id = d_scaned[myId];
  }
  d_outputVals[des_id] = d_inputVals[myId];
  d_outputPos[des_id] = d_inputPos[myId];
}

#ifdef USE_THRUST
void your_sort(unsigned int* const d_inputVals,
  unsigned int* const d_inputPos,
  unsigned int* const d_outputVals,
  unsigned int* const d_outputPos,
  const size_t numElems)
{
  // Thrust vectors wrapping raw GPU data
  thrust::device_ptr<unsigned int> d_inputVals_p(d_inputVals);
  thrust::device_ptr<unsigned int> d_inputPos_p(d_inputPos);
  thrust::host_vector<unsigned int> h_inputVals_vec(d_inputVals_p,
    d_inputVals_p + numElems);
  thrust::host_vector<unsigned int> h_inputPos_vec(d_inputPos_p,
    d_inputPos_p + numElems);
  // ?? device_vector is wrong
  thrust::sort_by_key(h_inputVals_vec.begin(), h_inputVals_vec.end(), h_inputPos_vec.begin());
  checkCudaErrors(cudaMemcpy(d_outputVals, thrust::raw_pointer_cast(&h_inputVals_vec[0]),
    numElems * sizeof(unsigned int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_outputPos, thrust::raw_pointer_cast(&h_inputPos_vec[0]),
    numElems * sizeof(unsigned int), cudaMemcpyHostToDevice));
}
#else
void your_sort(unsigned int* const d_inputVals,
  unsigned int* const d_inputPos,
  unsigned int* const d_outputVals,
  unsigned int* const d_outputPos,
  const size_t numElems)
{
  // use how many bits/time to compare(maybe 4 is most efficent)
  const int numBits = 1;  //??
  const int numBins = 1 << numBits;
  const int m = 1 << 10;
  int blocks = ceil((float)numElems / m);
  printf("m %d blocks %d\n", m ,blocks);
  // allocate GPU memory
  unsigned int *d_binHistogram;
  checkCudaErrors(cudaMalloc(&d_binHistogram, sizeof(unsigned int)* numBins));
  // not numBins --> different from CPU version
  thrust::device_vector<unsigned int> d_scan(numElems);

  // Loop bits: only guaranteed to work for numBits that are multiples of 2
  for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i++) {
    //unsigned int mask = 1 << i;
    checkCudaErrors(cudaMemset(d_binHistogram, 0, sizeof(unsigned int)* numBins));
    // 1) perform histogram of data & mask into bins
    histo_kernel << <blocks, m >> >(d_binHistogram, d_inputVals, i, numElems);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    //print_kernel << <1, 2 >> >(d_binHistogram);
    //printf("\n");
    // 2) perform exclusive prefix sum (scan) on binHistogram to get starting
    // location for each bin
    sumscan_kernel << <1, numBins, sizeof(unsigned int)* numBins>> >(d_binHistogram, numBins, numElems);
    //print_kernel << <1, 2 >> >(d_binHistogram);
    //printf("\n");
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // 3) Gather everything into the correct location
    // need to move vals and positions
    makescan_kernel << <blocks, m >> >(d_inputVals, thrust::raw_pointer_cast(&d_scan[0]), i, numElems);
    //print_kernel << <1, 4 >> >(thrust::raw_pointer_cast(&d_scan[0]));
    //printf("\n");
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // segmented scan described in http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
    //thrust::host_vector<unsigned int> h_scan = d_scan;
    //printf("%d %d %d\n", h_scan[0], h_scan[1], h_scan[2]);
    thrust::exclusive_scan(d_scan.begin(), d_scan.end(), d_scan.begin());
    //print_kernel << <1, 4 >> >(thrust::raw_pointer_cast(&d_scan[0]));
   // printf("\n");
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    //thrust::host_vector<unsigned int> h_scan_2 = d_scan;
    //printf("%d %d %d\n", h_scan_2[0], h_scan_2[1], h_scan_2[2]);
    move_kernel << <blocks, m >> >(d_inputVals, d_inputPos, d_outputVals, d_outputPos,
      numElems, d_binHistogram, thrust::raw_pointer_cast(&d_scan[0]), i);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_inputPos, d_outputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
  }
  // Free memory
  checkCudaErrors(cudaFree(d_binHistogram));
}
#endif