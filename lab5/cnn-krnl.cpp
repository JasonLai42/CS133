// If you want to modify the tiling size, uncomment:
// #define kTileH   (28)
// #define kTileW   (56)

// Tiling specification must be before the #include
// and 224 must be a multiple of the tiling size
#include "lib/cnn-krnl.h"

void CnnKernel_YourCode(
    const input_g_t *input_g, const weight_g_t *weight_g,
    const bias_g_t  *bias_g,        output_g_t *output_g) {

  static input_t   input [kNum][kTileH+kKernel-1][kTileW+kKernel-1];
  static weight_t  weight[kNum][kNum][kKernel][kKernel];
  static bias_t    bias  [kNum];
  static output_t  output[kNum][kTileH/2][kTileW/2];

  static compute_t C[kTileH][kTileW];

  int k = -1;
  int old_k = 0;
  input_t input_window[kKernel-1][kKernel];
  #pragma HLS array_partition variable=input_window dim=0 complete

  // TODO:  You may want to add array partitioning here, e.g.:
  #pragma HLS array_partition variable=bias factor=56 cyclic
  #pragma HLS array_partition variable=weight dim=3 factor=5 cyclic
  #pragma HLS array_partition variable=input dim=3 factor=5 cyclic

  // Read the whole arrays from memory to device
  read_weight_from_memory(weight_g, weight);
  read_bias_from_memory  (bias_g,   bias);

  main_loop_tile_h:
  for (int hh = 0; hh < kImSize; hh += kTileH) {

    main_loop_tile_w:
    for (int ww = 0; ww < kImSize; ww += kTileW) {

      // Read input[j][h][w] = Input(j, hh + h, ww + w);
      read_input_from_memory(hh, ww, input_g, input);

      main_loop_i:
      for (int i = 0; i < kNum; ++i) {
        // TODO:  Please modify the code inside this loop :-)

        // You can use printf in software simulation for debugging
        fprintf(stderr, "Finished %d%% channel(s) #%d/#%d\r",
                100*i/kNum, i, kNum);

        // Set bias
        set_bias:
        for (int h = 0; h < kTileH; ++h) {
          for (int w = 0; w < kTileW; ++w) {
            C[h][w] = bias[i];
          }
        }

        // Convolution
        conv:
        for (int j = 0; j < kNum; ++j) {
          // Initialize sliding window for this channel j
          for (int u = 0; u < kKernel-1; ++u) {
            for (int v = 0; v < kKernel; ++v) {
              input_window[u][v] = input[j][u][v];
            }
          }
          for (int h = 0; h < kTileH; ++h) {
            for (int w = 0; w < kTileW; ++w) {
              for (int p = 0; p < kKernel; ++p) {
                k++;
                for (int q = 0; q < kKernel; ++q) {
                  if(p != 4) {
                    C[h][w] += weight[i][j][p][q] *
                               input_window[k][q];
                  }
                  else {
                    C[h][w] += weight[i][j][p][q] *
                               input[j][h + p][w + q];
                    input_window[old_k][q] = input[j][h + p][w + q];
                  }
                }
              }
              old_k = (old_k + 1) % 4;
              k = old_k;
            }
          }
        }

        // ReLU
        relu:
        for (int h = 0; h < kTileH; ++h) {
          for (int w = 0; w < kTileW; ++w) {
            if (C[h][w] < 0) C[h][w] = 0;
          }
        }

        // Max pooling
        maxpool:
        for (int h = 0; h < kTileH/2; ++h) {
          for (int w = 0; w < kTileW/2; ++w) {
            output[i][h][w] = max(
                max(C[h * 2][w * 2    ], C[h * 2 + 1][w * 2    ]),
                max(C[h * 2][w * 2 + 1], C[h * 2 + 1][w * 2 + 1]));
          }
        }
      }

      // Write Output(i, hh/2 + h, ww/2 + w) = output[i][h][w];
      write_output_to_memory(hh, ww, output_g, output);

      fprintf(stderr, "Computation for tile (%d, %d) is completed.\n",
              hh, ww) ;
    }
  }

}
