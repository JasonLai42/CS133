// Header inclusions, if any...

#include <cstring>
#include <omp.h>

#include "lib/gemm.h"

// Using declarations, if any...

void GemmParallel(const float a[kI][kK], const float b[kK][kJ],
                  float c[kI][kJ]) {
  // Your code goes here...
  for (int i = 0; i < kI; ++i) {
    std::memset(c[i], 0, sizeof(float) * kJ);
  }
  for (int k = 0; k < kK; k += 4) {
    for (int i = 0; i < kI; ++i) {
      for (int j = 0; j < kJ; ++j) {
        c[i][j] += a[i][k] * b[k][j];
        c[i][j] += a[i][k+1] * b[k+1][j];
        c[i][j] += a[i][k+2] * b[k+2][j];
        c[i][j] += a[i][k+3] * b[k+3][j];
      }
    }
  }
}
