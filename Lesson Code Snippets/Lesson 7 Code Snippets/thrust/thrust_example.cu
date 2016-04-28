#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>
#include <vector>

#include "gputimer.h"
#include "gettime.h"

int main(void)
{
  // generate N random numbers serially
  int N = 1000000;
  std::vector<char> h_vec(N);
  std::generate(h_vec.begin(), h_vec.end(), rand);
  std::vector<char> h_vec_std(h_vec);

  double t0 = tic();
  thrust::sort(h_vec.begin(), h_vec.end());
  std::cout << "thrust::sort took " << tic() - t0 << " ms." << std::endl;

  t0 = tic();
  std::sort(h_vec_std.begin(), h_vec_std.end());
  std::cout << "std::sort took " << tic() - t0 << " ms." << std::endl;

  for (int i = 0; i < N; i++) {
    if (h_vec[i] != h_vec_std[i]) {
      std::cout << i << " Not same!" << std::endl;
      exit(1);
    }
  }

  return 0;
}