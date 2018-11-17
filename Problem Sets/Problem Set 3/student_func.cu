/* Udacity Homework 3
HDR Tone-mapping

Background HDR
==============

A High Dynamic Range (HDR) image contains a wider variation of intensity
and color than is allowed by the RGB format with 1 byte per channel that we
have used in the previous assignment.

To store this extra information we use single precision floating point for
each channel.  This allows for an extremely wide range of intensity values.

In the image for this assignment, the inside of church with light coming in
through stained glass windows, the raw input floating point values for the
channels range from 0 to 275.  But the mean is .41 and 98% of the values are
less than 3!  This means that certain areas (the windows) are extremely bright
compared to everywhere else.  If we linearly map this [0-275] range into the
[0-255] range that we have been using then most values will be mapped to zero!
The only thing we will be able to see are the very brightest areas - the
windows - everything else will appear pitch black.

The problem is that although we have cameras capable of recording the wide
range of intensity that exists in the real world our monitors are not capable
of displaying them.  Our eyes are also quite capable of observing a much wider
range of intensities than our image formats / monitors are capable of
displaying.

Tone-mapping is a process that transforms the intensities in the image so that
the brightest values aren't nearly so far away from the mean.  That way when
we transform the values into [0-255] we can actually see the entire image.
There are many ways to perform this process and it is as much an art as a
science - there is no single "right" answer.  In this homework we will
implement one possible technique.

Background Chrominance-Luminance
================================

The RGB space that we have been using to represent images can be thought of as
one possible set of axes spanning a three dimensional space of color.  We
sometimes choose other axes to represent this space because they make certain
operations more convenient.

Another possible way of representing a color image is to separate the color
information (chromaticity) from the brightness information.  There are
multiple different methods for doing this - a common one during the analog
television days was known as Chrominance-Luminance or YUV.

We choose to represent the image in this way so that we can remap only the
intensity channel and then recombine the new intensity values with the color
information to form the final image.

Old TV signals used to be transmitted in this way so that black & white
televisions could display the luminance channel while color televisions would
display all three of the channels.


Tone-mapping
============

In this assignment we are going to transform the luminance channel (actually
the log of the luminance, but this is unimportant for the parts of the
algorithm that you will be implementing) by compressing its range to [0, 1].
To do this we need the cumulative distribution of the luminance values.

Example
-------

input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
min / max / range: 0 / 9 / 9

histo with 3 bins: [4 7 3]
// exclusive
cdf : [0 4 11]


Your task is to calculate this cumulative distribution by following these
steps.

*/

#include "utils.h"
#include<device_launch_parameters.h>
#include<device_functions.h>

__global__ void shmem_reduce_kernel(float * d_out, const float * const d_in, bool is_max)
{
  // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
  extern __shared__ float sdata[];

  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  // load shared mem from global mem
  sdata[tid] = d_in[myId];
  __syncthreads();            // make sure entire block is loaded!

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (tid < s)
    {
      if (is_max)
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

__global__ void histo_kernel(unsigned int * d_out, const float * const d_in,
  const size_t numBins, float logLumRange, float min_logLum)
{
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int bin = (d_in[myId] - min_logLum) / logLumRange * numBins;
  if (bin == numBins)  bin--;
  atomicAdd(&d_out[bin], 1);
}

__global__ void scan_kernel(unsigned int * d_out, const float * const d_in,
  const size_t numBins, float logLumRange, float min_logLum)
{
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int bin = (d_in[myId] - min_logLum) / logLumRange * numBins;
  if (bin == numBins)  bin--;
  atomicAdd(&d_out[bin], 1);
}

// Hillis Steele Scan - described in lecture
__global__ void cdf_kernel(unsigned int * d_in, const size_t numBins)
{
  int myId = threadIdx.x;
  for (int d = 1; d < numBins; d *= 2) {
    if ((myId + 1) % (d * 2) == 0) {
      d_in[myId] += d_in[myId - d];
    }
    __syncthreads();
  }
  if (myId == numBins - 1) d_in[myId] = 0;
  for (int d = numBins / 2; d >= 1; d /= 2) {
    if ((myId + 1) % (d * 2) == 0) {
      unsigned int tmp = d_in[myId - d];
      d_in[myId - d] = d_in[myId];
      d_in[myId] += tmp;
    }
    __syncthreads();
  }
}

// Blelloch Scan - described in lecture
__global__ void cdf_kernel_2(unsigned int * d_in, const size_t numBins)
{ 
  int idx = threadIdx.x;
  extern __shared__ int temp[];
  int pout = 0, pin = 1;

  temp[idx] = (idx > 0) ? d_in[idx - 1] : 0;
  __syncthreads();

  for (int offset = 1; offset < n; offset *= 2) {
    // swap double buffer indices
    pout = 1 - pout;
    pin = 1 - pout;
    if (idx >= offset) {
      temp[pout*n+idx] = temp[pin*n+idx - offset] + temp[pin*n+idx];  // changed line
    } else {
      temp[pout*n+idx] = temp[pin*n+idx];
    }
    __syncthreads();
  }
  d_in[idx] = temp[pout*n+idx];
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
  unsigned int* const d_cdf,
  float &min_logLum,
  float &max_logLum,
  const size_t numRows,
  const size_t numCols,
  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
  1) find the minimum and maximum value in the input logLuminance channel
  store in min_logLum and max_logLum */
  // use reduce
  const int m = 1 << 10;
  int blocks = ceil((float)numCols * numRows / m);

  float *d_intermediate; // should not modify d_in
  checkCudaErrors(cudaMalloc(&d_intermediate, sizeof(float)* blocks)); // store max and min
  float *d_min, *d_max;
  checkCudaErrors(cudaMalloc((void **)&d_min, sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_max, sizeof(float)));

  shmem_reduce_kernel << <blocks, m, m * sizeof(float) >> >(d_intermediate, d_logLuminance, true);
  shmem_reduce_kernel << <1, blocks, blocks * sizeof(float) >> >(d_max, d_intermediate, true);
  shmem_reduce_kernel << <blocks, m, m * sizeof(float) >> >(d_intermediate, d_logLuminance, false);
  shmem_reduce_kernel << <1, blocks, blocks * sizeof(float) >> >(d_min, d_intermediate, false);
  checkCudaErrors(cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_intermediate));
  checkCudaErrors(cudaFree(d_min));
  checkCudaErrors(cudaFree(d_max));

  //2) subtract them to find the range
  float logLumRange = max_logLum - min_logLum;
  printf("max_logLum: %f  min_logLum: %f  logLumRange: %f\n", max_logLum, min_logLum, logLumRange);

  //3) generate a histogram of all the values in the logLuminance channel using
  //   the formula: bin = (lum[i] - lumMin) / lumRange * numBins
  checkCudaErrors(cudaMemset(d_cdf, 0, sizeof(unsigned int)* numBins));
  histo_kernel << <blocks, m >> >(d_cdf, d_logLuminance, numBins, logLumRange, min_logLum);

  //4) Perform an exclusive scan (prefix sum) on the histogram to get
  //   the cumulative distribution of luminance values (this should go in the
  //   incoming d_cdf pointer which already has been allocated for you)      
  //cdf_kernel << <1, numBins>> >(d_cdf, numBins);
  cdf_kernel_2 << <1, numBins, sizeof(unsigned int) * numBins * 2 >> >(d_cdf, numBins);
}
