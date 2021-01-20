// Header inclusions, if any...

#include <cstring>
#include <omp.h>

#include "lib/gemm.h"

// Using declarations, if any...

void GemmParallel(const float a[kI][kK], const float b[kK][kJ],
                  float c[kI][kJ]) {
  // Your code goes here...

  #pragma omp parallel for num_threads(8)
  for (int i = 0; i < kI; ++i) {
    std::memset(c[i], 0, sizeof(float) * kJ);
  }

  int k, i, j;
  for (k = 0; k < kK; k += 4) {
    #pragma omp parallel for num_threads(8) private(j) schedule(guided)
    for (i = 0; i < kI; ++i) {
      for (j = 0; j < kJ; ++j) {
        c[i][j] += (a[i][k] * b[k][j]) + (a[i][k+1] * b[k+1][j]) + (a[i][k+2] * b[k+2][j]) + (a[i][k+3] * b[k+3][j]);
      }
    }
  }
}
