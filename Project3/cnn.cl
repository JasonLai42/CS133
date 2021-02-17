__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;

#define c_access(x,y,z) C[(x)*kImSize*kImSize+(y)*kImSize+(z)]
#define weight_access(w,x,y,z) weight[(w)*kNum*kKernel*kKernel+(x)*kKernel*kKernel+(y)*kKernel+(z)]
#define input_access(x,y,z) input[(x)*kImSize*kImSize+(y)*kImSize+(z)]
#define output_access(x,y,z) output[(x)*kOutImSize*kOutImSize+(y)*kOutImSize+(z)]

__kernel
void CnnKernel(__constant float* input, __constant float* weight,
               __constant float* bias, __global float* output) {
  // your code goes here

  // LOOP 1: Set bias for each channel
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w)
        // C[i][h][w] = bias[i];
        c_access(i,h,w) = bias[i];
    }
  }

  // LOOP 2: Convolution
  for (int i = 0; i < kNum; ++i) { // Which of the 256 filters we're on
    for (int j = 0; j < kNum; ++j) { // Which of the 256 channels we're on
      for (int h = 0; h < kImSize; ++h) { // Which row of the 224 x 224 intermediate image we're on
        for (int w = 0; w < kImSize; ++w) { // Which column of the 224 x 224 intermediate image we're on
          for (int p = 0; p < kKernel; ++p) { // Which row of the window/filter we're on
            for (int q = 0; q < kKernel; ++q) // Which column of the window/filter we're on
              // C[i][h][w] += weight[i][j][p][q] * input[j][h + p][w + q];
              c_access(i,h,w) += weight(i,j,p,q) * input(j,(h + p),(w + q));
          }
        }
      }
    }
  }

  // LOOP 3: ReLU
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w) {
        // C[i][h][w] = max(0.f, C[i][h][w]);
        c_access(i,h,w) = max(0.f, c_access(i,h,w));
      }
    }
  }

  // LOOP 4: Max pooling
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kOutImSize; ++h) {
      for (int w = 0; w < kOutImSize; ++w) {
        // output[i][h][w] = max(
          // max(C[i][h * 2][w * 2    ], C[i][h * 2 + 1][w * 2    ]),
          // max(C[i][h * 2][w * 2 + 1], C[i][h * 2 + 1][w * 2 + 1]));
        output_access(i,h,w) = max(
          max(c_access(i,(h * 2),(w * 2    )), c_access(i,(h * 2 + 1),(w * 2    )))
          max(c_access(i,(h * 2),(w * 2 + 1)), c_access(i,(h * 2 + 1),(w * 2 + 1))));
      }
    }
  }
}
