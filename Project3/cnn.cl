__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;

#define c_access(y,z) C[(y)*kInImSize+(z)]
#define weight_access(w,x,y,z) weight[(w)*kNum*kKernel*kKernel+(x)*kKernel*kKernel+(y)*kKernel+(z)]
#define input_access(x,y,z) input[(x)*kImSize*kImSize+(y)*kImSize+(z)]
#define output_access(x,y,z) output[(x)*kOutImSize*kOutImSize+(y)*kOutImSize+(z)]

__kernel
void CnnKernel(__constant float* input, __constant float* weight,
               __constant float* bias, __global float* output) {
  // your code goes here

  // const int gGR = get_group_id(0), gGC = get_group_id(1);
  // const int lR = get_local_id(0), lC = get_local_id(1);
  // const int row = gGR*kImSize+lR, col = gGC*kImSize+lC;

  // Intermediate image array
  __local float c_access(kImSize,kImSize);

  for (int i = 0; i < kNum; ++i) { // Which of the 256 filters we're on
    // LOOP 1: Set bias for each channel
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w)
        c_access(h,w) = bias[i];
    }


    // LOOP 2: Convolution
    for (int j = 0; j < kNum; ++j) { // Which of the 256 channels we're on
      for (int h = 0; h < kImSize; ++h) { // Which row of the 224 x 224 intermediate image we're on
        for (int w = 0; w < kImSize; ++w) { // Which column of the 224 x 224 intermediate image we're on
          for (int p = 0; p < kKernel; ++p) { // Which row of the window/filter we're on
            for (int q = 0; q < kKernel; ++q) // Which column of the window/filter we're on
              c_access(h,w) += weight_access(i,j,p,q) * input_access(j,(h + p),(w + q));
          }
        }
      }
    }


    // LOOP 3: ReLU
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w) {
        c_access(h,w) = max(0.f, c_access(h,w));
      }
    }


    // LOOP 4: Max pooling
    for (int h = 0; h < kOutImSize; ++h) {
      for (int w = 0; w < kOutImSize; ++w) {
        output_access(i,h,w) = max(
          max(c_access((h * 2),(w * 2    )), c_access((h * 2 + 1),(w * 2    ))), 
          max(c_access((h * 2),(w * 2 + 1)), c_access((h * 2 + 1),(w * 2 + 1))));
      }
    }
  }
}
