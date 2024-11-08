// If you want to modify the tiling size, uncomment:
#define kTileH   (112)
#define kTileW   (56)

// Tiling specification must be before the #include
// and 224 must be a multiple of the tiling size
#include "lib/cnn-krnl.h"

void InitWindow(input_t (&window)[kKernel][kTileW + kKernel - 1], input_t (&array)[kTileH+kKernel-1][kTileW+kKernel-1]) {
  #pragma HLS inline off
    init_window:
    for (int u = 0; u < kKernel; ++u) {
      #pragma HLS pipeline
      for (int v = 0; v < kTileW + kKernel - 1; ++v) {
        window[u][v] = array[u][v];
      }
    }
}

void CnnKernel_YourCode(
    const input_g_t *input_g, const weight_g_t *weight_g,
    const bias_g_t  *bias_g,        output_g_t *output_g) {

  static input_t   input [kNum][kTileH+kKernel-1][kTileW+kKernel-1];
  static weight_t  weight[kNum][kNum][kKernel][kKernel];
  static bias_t    bias  [kNum];
  static output_t  output[kNum][kTileH/2][kTileW/2];

  static compute_t C[kTileH][kTileW];

  // FOR SLIDING WINDOW
  input_t input_window[kKernel][kTileW + kKernel - 1];
  #pragma HLS array_partition variable=input_window dim=0 complete

  // TODO:  You may want to add array partitioning here, e.g.:
  #pragma HLS array_partition variable=bias factor=16 cyclic
  #pragma HLS array_partition variable=weight dim=3 factor=5 cyclic
  #pragma HLS array_partition variable=input dim=2 factor=5 cyclic
  #pragma HLS array_partition variable=input dim=3 complete

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
        set_bias_h:
        for (int h = 0; h < kTileH; ++h) {
          #pragma HLS pipeline
          set_bias_w:
          for (int w = 0; w < kTileW; ++w) {
            C[h][w] = bias[i];
          }
        }

        // Convolution
        conv_j:
        for (int j = 0; j < kNum; ++j) {
          InitWindow(input_window, input[j]);
          conv_h:
          for (int h = 0; h < kTileH; ++h) {
            #pragma HLS pipeline
            conv_w:
            for (int w = 0; w < kTileW; ++w) {
              conv_p:
              for (int p = 0; p < kKernel; ++p) {
                conv_q:
                for (int q = 0; q < kKernel; ++q) {
                  C[h][w] += weight[i][j][p][q] * 
                             input_window[p][w + q];

                  if(p == 4 && (q == 0 || w == (kTileW - 1))) {
                    conv_shift:
                    for (int t = 0; t < kKernel - 1; ++t) {
                      input_window[t][w + q] = input_window[t + 1][w + q];
                    }
                    input_window[4][w + q] = input[j][h + p + 1][w + q];
                  }
                }
              }
            }
          }
        }

        // ReLU
        relu_h:
        for (int h = 0; h < kTileH; ++h) {
          #pragma HLS pipeline
          relu_w:
          for (int w = 0; w < kTileW; ++w) {
            if (C[h][w] < 0) C[h][w] = 0;
          }
        }

        // Max pooling
        maxpool_h:
        for (int h = 0; h < kTileH/2; ++h) {
          #pragma HLS pipeline
          maxpool_w:
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
