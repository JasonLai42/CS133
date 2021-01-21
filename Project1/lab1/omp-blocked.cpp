// Header inclusions, if any...

#include <cstring>
#include <omp.h>

#include "lib/gemm.h"

// Using declarations, if any...

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
  // Your code goes here...
  
  int block_sizeI = 32;
  int block_sizeJ = 512;
  int block_sizeK = 256;

  #pragma omp parallel for num_threads(8)
  for (int i = 0; i < kI; ++i) {
    std::memset(c[i], 0, sizeof(float) * kJ);
  }

  int i, j, k, ii, jj, kk;
  #pragma omp parallel for num_threads(8) private(k, j, kk, ii, jj)
  for (i = 0; i < kI; i += block_sizeI) {
    for (k = 0; k < kK; k += block_sizeK) {
      for (j = 0; j < kJ; j += block_sizeJ) {
        for(kk = k; kk < k + block_sizeK; kk += 4) {
          for(ii = i; ii < i + block_sizeI; ++ii) {
            for(jj = j; jj < j + block_sizeJ; ++jj) {
              c[ii][jj] += (a[ii][kk] * b[kk][jj]) 
                        + (a[ii][kk+1] * b[kk+1][jj]) 
                        + (a[ii][kk+2] * b[kk+2][jj]) 
                        + (a[ii][kk+3] * b[kk+3][jj]);
            }
          }
        }
      }
    }
  }
}
