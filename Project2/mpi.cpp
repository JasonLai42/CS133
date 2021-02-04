// Header inclusions, if any...

#include <string.h>
#include <mpi.h>

#include "lib/gemm.h"
#include "lib/common.h"
// You can directly use aligned_alloc
// with lab2::aligned_alloc(...)

// Using declarations, if any...

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
  // Your code goes here...

  // Tile sizes
  int block_sizeI = 32;
  int block_sizeJ = 512;
  int block_sizeK = 128;

  // Get ranks and number of processes
  int rank, num_proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

  int chunk_size = kI / num_proc;

  // Send buffers
  float *sendA = (float*)lab2::aligned_alloc(4096, kI * kK * sizeof(float));
  float *sendB = (float*)lab2::aligned_alloc(4096, kK * kJ * sizeof(float));
  if(rank == 0) {
    memcpy(sendA, a, kI * kK * sizeof(float));
    memcpy(sendB, b, kK * kJ * sizeof(float));
  }
  
  // Store computed rows of c
  float *productC = (float*)lab2::aligned_alloc(4096, chunk_size * kJ * sizeof(float));

  // Receive buffers
  float *recA = (float*)lab2::aligned_alloc(4096, chunk_size * kK * sizeof(float));

  // Scatter allocated rows of matrix a to specific ranked processes; Broadcast b matrix to all processes
  MPI_Scatter(sendA, chunk_size * kK, MPI_FLOAT, recA, chunk_size * kK, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(sendB, kK * kJ, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Set all values of pieces of matrix c result to 0
  memset(productC, 0, chunk_size * kJ * sizeof(float));

  // Only iterate up to chunk_size rows each "system" gets allocated
  for (int i = 0; i < chunk_size; i += block_sizeI) {
    for (int k = 0; k < kK; k += block_sizeK) {
      for (int j = 0; j < kJ; j += block_sizeJ) {
        for(int kk = k; kk < k + block_sizeK; kk += 4) {
          for(int ii = i; ii < i + block_sizeI; ++ii) {
            for(int jj = j; jj < j + block_sizeJ; ++jj) {
              productC[(ii * kJ) + jj] += (recA[(ii * kK) + kk] * sendB[(kk * kJ) + jj]) 
                        + (recA[(ii * kK) + kk + 1] * sendB[((kk + 1) * kJ) + jj]) 
                        + (recA[(ii * kK) + kk + 2] * sendB[((kk + 2) * kJ) + jj]) 
                        + (recA[(ii * kK) + kk + 3] * sendB[((kk + 3) * kJ) + jj]);
            }
          }
        }
      }
    }
  }

  // Gather chunks back to process 0
  MPI_Gather(productC, chunk_size * kJ, MPI_FLOAT, c, chunk_size * kJ, MPI_FLOAT, 0, MPI_COMM_WORLD);
}