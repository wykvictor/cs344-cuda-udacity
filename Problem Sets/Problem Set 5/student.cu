/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

__global__
void atomic_kernel(const unsigned int* const d_vals, //INPUT
               unsigned int* const d_histo,      //OUPUT
               const unsigned int numElems)
{
  int myId = threadIdx.x + blockIdx.x * blockDim.x;
  if (myId >= numElems) return;
  atomicAdd(&d_histo[d_vals[myId]], 1);
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems, int types)
{
  const int m = 1 << 10;
  int blocks = ceil((float)numElems / m);

  /*thrust::device_ptr<unsigned int> in_vals(d_vals);
  thrust::device_ptr<unsigned int> in_keys(d_vals);
  thrust::device_ptr<unsigned int> out_vals(d_histo);
  unsigned int* out_keys;
  checkCudaErrors(cudaMalloc(&out_keys, sizeof(unsigned int)*numElems));*/

  switch (types){
  case 0:
    atomic_kernel << <blocks, m >> >(d_vals, d_histo, numElems);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    break;

  // https://www.ecse.rpi.edu/~wrf/wiki/ParallelComputingSpring2014/thrust/histogram.cu
  case 1:
    //thrust::sort(in_vals, in_vals + numElems);
    //thrust::reduce_by_key(in_keys, in_keys + numElems, in_vals, out_keys, out_vals);
    
    break;
  case 3:
    break;
  }
}
