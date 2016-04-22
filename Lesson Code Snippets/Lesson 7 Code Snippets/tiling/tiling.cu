#include <stdio.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include "gputimer.h"
#include "utils.h"

const int BLOCKSIZE	= 128;
const int NUMBLOCKS = 100;					// set this to 1 or 2 for debugging
const int N 		= BLOCKSIZE*NUMBLOCKS;

/* 
 * TODO: modify the foo and bar kernels to use tiling: 
 * 		 - copy the input data to shared memory
 *		 - perform the computation there
 *	     - copy the result back to global memory
 *		 - assume thread blocks of 128 threads
 *		 - handle intra-block boundaries correctly
 * You can ignore boundary conditions (we ignore the first 2 and last 2 elements)
 */
__global__ void foo(float out[], float A[], float B[], float C[], float D[], float E[]){

	int i = threadIdx.x + blockIdx.x*blockDim.x; 
	
	out[i] = (A[i] + B[i] + C[i] + D[i] + E[i]) / 5.0f;
}

__global__ void bar(float out[], float in[]) 
{
	int i = threadIdx.x + blockIdx.x*blockDim.x; 

	out[i] = (in[i-2] + in[i-1] + in[i] + in[i+1] + in[i+2]) / 5.0f;
}

__global__ void bar_tile(float out[], float in[])
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int idx = threadIdx.x;
  extern __shared__ float sh_din[];  
  sh_din[idx + 2] = in[i];
  if (idx == 0) {
    sh_din[idx] = in[i-2];
    sh_din[idx+1] = in[i-1];
  }
  else if (idx == blockDim.x - 1) {
    sh_din[idx + 3] = in[i+1];
    sh_din[idx + 4] = in[i+2];
  }
  __syncthreads();

  out[i] = (sh_din[idx] + sh_din[idx + 1] + sh_din[idx + 2] + sh_din[idx + 3] + sh_din[idx + 4]) / 5.0f;
}

__global__ void bar_tile_2(float out[], float in[])
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int idx = threadIdx.x;
  extern __shared__ float sh_din[];
  sh_din[idx] = in[i];
  __syncthreads();
  if (idx == 0) {
    out[i] = (in[i - 2] + in[i - 1] + sh_din[idx] + sh_din[idx + 1] + sh_din[idx + 2]) / 5.0f;
  }
  else if (idx == 1) {
    out[i] = (in[i - 2] + sh_din[idx - 1] + sh_din[idx] + sh_din[idx + 1] + sh_din[idx + 2]) / 5.0f;
  }
  else if (idx == blockDim.x - 2) {
    out[i] = (sh_din[idx - 2] + sh_din[idx - 1] + sh_din[idx] + sh_din[idx + 1] + in[i + 2]) / 5.0f;
  }
  else if (idx == blockDim.x - 1) {
    out[i] = (sh_din[idx - 2] + sh_din[idx - 1] + sh_din[idx] + in[i + 1] + in[i + 2]) / 5.0f;
  }
  else {
    out[i] = (sh_din[idx - 2] + sh_din[idx - 1] + sh_din[idx] + sh_din[idx + 1] + sh_din[idx + 2]) / 5.0f;
  }  
}

__global__ void bar_tile_3(float out[], float in[])
{
  int idx = threadIdx.x;
  extern __shared__ float sh_din[];
  int i_in = blockIdx.x * BLOCKSIZE + idx;
  sh_din[idx] = in[i_in-2];
  __syncthreads();
  if (idx < blockDim.x-4)
    out[i_in] = (sh_din[idx] + sh_din[idx + 1] + sh_din[idx + 2] + sh_din[idx + 3] + sh_din[idx + 4]) / 5.0f;
}

void cpuFoo(float out[], float A[], float B[], float C[], float D[], float E[])
{
	for (int i=0; i<N; i++)
	{
		out[i] = (A[i] + B[i] + C[i] + D[i] + E[i]) / 5.0f;
	}
}

void cpuBar(float out[], float in[])
{
	// ignore the boundaries
	for (int i=2; i<N-2; i++)
	{
		out[i] = (in[i-2] + in[i-1] + in[i] + in[i+1] + in[i+2]) / 5.0f;
	}
}

int main(int argc, char **argv)
{
	// declare and fill input arrays for foo() and bar()
	float fooA[N], fooB[N], fooC[N], fooD[N], fooE[N], barIn[N];
	for (int i=0; i<N; i++) 
	{
		fooA[i] = i; 
		fooB[i] = i+1;
		fooC[i] = i+2;
		fooD[i] = i+3;
		fooE[i] = i+4;
		barIn[i] = 2*i; 
	}
	// device arrays
	int numBytes = N * sizeof(float);
	float *d_fooA;	 	checkCudaErrors(cudaMalloc(&d_fooA, numBytes));
	float *d_fooB; 		checkCudaErrors(cudaMalloc(&d_fooB, numBytes));
	float *d_fooC;	 	checkCudaErrors(cudaMalloc(&d_fooC, numBytes));
	float *d_fooD; 		checkCudaErrors(cudaMalloc(&d_fooD, numBytes));
	float *d_fooE; 		checkCudaErrors(cudaMalloc(&d_fooE, numBytes));
	float *d_barIn; 	checkCudaErrors(cudaMalloc(&d_barIn, numBytes));
	checkCudaErrors(cudaMemcpy(d_fooA, fooA, numBytes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_fooB, fooB, numBytes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_fooC, fooC, numBytes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_fooD, fooD, numBytes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_fooE, fooE, numBytes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_barIn, barIn, numBytes, cudaMemcpyHostToDevice));	

	// output arrays for host and device
	float fooOut[N], barOut[N], *d_fooOut, *d_barOut;
	checkCudaErrors(cudaMalloc(&d_fooOut, numBytes));
	checkCudaErrors(cudaMalloc(&d_barOut, numBytes));

	// declare and compute reference solutions
	float ref_fooOut[N], ref_barOut[N]; 
	cpuFoo(ref_fooOut, fooA, fooB, fooC, fooD, fooE);
	cpuBar(ref_barOut, barIn);

	// launch and time foo and bar
	GpuTimer fooTimer, barTimer;
	fooTimer.Start();
	foo<<<N/BLOCKSIZE, BLOCKSIZE>>>(d_fooOut, d_fooA, d_fooB, d_fooC, d_fooD, d_fooE);
	fooTimer.Stop();
	
  checkCudaErrors(cudaMemcpy(fooOut, d_fooOut, numBytes, cudaMemcpyDeviceToHost));
  printf("foo<<<>>>(): %g ms elapsed. Verifying solution...", fooTimer.Elapsed());
  compareArrays(ref_fooOut, fooOut, N);

	barTimer.Start();
	bar<<<N/BLOCKSIZE, BLOCKSIZE>>>(d_barOut, d_barIn);
  //bar_tile << <N / BLOCKSIZE, BLOCKSIZE, (BLOCKSIZE + 4) * sizeof(float) >> >(d_barOut, d_barIn);
  //bar_tile_2 << <N / BLOCKSIZE, BLOCKSIZE, (BLOCKSIZE) * sizeof(float) >> >(d_barOut, d_barIn);
  //bar_tile_3 << <N / BLOCKSIZE, BLOCKSIZE+4, (BLOCKSIZE + 4) * sizeof(float) >> >(d_barOut, d_barIn);
	barTimer.Stop();

	checkCudaErrors(cudaMemcpy(barOut, d_barOut, numBytes, cudaMemcpyDeviceToHost));
	printf("bar<<<>>>(): %g ms elapsed. Verifying solution...", barTimer.Elapsed());
	compareArrays(ref_barOut, barOut, N);

  barTimer.Start();
  bar_tile << <N / BLOCKSIZE, BLOCKSIZE, (BLOCKSIZE + 4) * sizeof(float) >> >(d_barOut, d_barIn);
  barTimer.Stop();

  checkCudaErrors(cudaMemcpy(barOut, d_barOut, numBytes, cudaMemcpyDeviceToHost));
  printf("bar_tile<<<>>>(): %g ms elapsed. Verifying solution...", barTimer.Elapsed());
  compareArrays(ref_barOut, barOut, N);

  barTimer.Start();
  bar_tile_2 << <N / BLOCKSIZE, BLOCKSIZE, (BLOCKSIZE)* sizeof(float) >> >(d_barOut, d_barIn);
  barTimer.Stop();

  checkCudaErrors(cudaMemcpy(barOut, d_barOut, numBytes, cudaMemcpyDeviceToHost));
  printf("bar_tile_2<<<>>>(): %g ms elapsed. Verifying solution...", barTimer.Elapsed());
  compareArrays(ref_barOut, barOut, N);

  barTimer.Start();
  bar_tile_3 << <N / BLOCKSIZE, BLOCKSIZE + 4, (BLOCKSIZE + 4) * sizeof(float) >> >(d_barOut, d_barIn);
  barTimer.Stop();

  checkCudaErrors(cudaMemcpy(barOut, d_barOut, numBytes, cudaMemcpyDeviceToHost));
  printf("bar_tile_3<<<>>>(): %g ms elapsed. Verifying solution...", barTimer.Elapsed());
  compareArrays(ref_barOut, barOut, N);

	checkCudaErrors(cudaFree(d_fooA));
	checkCudaErrors(cudaFree(d_fooB));
	checkCudaErrors(cudaFree(d_fooC));
	checkCudaErrors(cudaFree(d_fooD));
	checkCudaErrors(cudaFree(d_fooE));
  checkCudaErrors(cudaFree(d_barIn));
	checkCudaErrors(cudaFree(d_fooOut));
	checkCudaErrors(cudaFree(d_barOut));
}
