// Header inclusions, if any...

#include <cstring>
#include <omp.h>

#include "lib/gemm.h"

// Using declarations, if any...

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
  // Your code goes here...
  
  int block_size = 128;

  #pragma omp parallel for num_threads(8)
  for (int i = 0; i < kI; ++i) {
    std::memset(c[i], 0, sizeof(float) * kJ);
  }

  int i, j, k, ii, jj;
  #pragma omp parallel for num_threads(8) schedule(guided)
  for (i = 0; i < kI; i += block_size) {
    for (j = 0; j < kJ; j += block_size) {
      for (k = 0; k < kK; k += 4) {
        for(ii = i; ii < i + block_size; ++ii) {
          for(jj = j; jj < j + block_size; ++jj) {
            c[ii][jj] += a[ii][k] * b[k][jj];
            c[ii][jj] += a[ii][k+1] * b[k+1][jj];
            c[ii][jj] += a[ii][k+2] * b[k+2][jj];
            c[ii][jj] += a[ii][k+3] * b[k+3][jj];
          }
        }
      }
    }
  }
}
