__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;

#define c_access(y,z) C[(y)*kImSize+(z)]
#define d_access(y,z) D[(y)*kImSize+(z)]
#define weight_access(w,x,y,z) weight[(w)*kNum*kKernel*kKernel+(x)*kKernel*kKernel+(y)*kKernel+(z)]
#define input_access(x,y,z) input[(x)*kInImSize*kInImSize+(y)*kInImSize+(z)]
#define output_access(x,y,z) output[(x)*kOutImSize*kOutImSize+(y)*kOutImSize+(z)]

__kernel
void CnnKernel(__constant float* input, __constant float* weight,
               __constant float* bias, __global float* output) {
  // your code goes here

  // Tile size to prevent out of resource
  const int i_tile = 16;
  const int h_tile = 8;
  const int w_tile = 32;

  // const int gGR = get_group_id(0), gGC = get_group_id(1);
  // const int lR = get_local_id(0), lC = get_local_id(1);
  // const int row = gGR*kImSize+lR, col = gGC*kImSize+lC;

  // Intermediate image array
  __local float c_access(h_tile,w_tile);
  __local float d_access(h_tile,w_tile);

for (int h = 0; h < kImSize; h += h_tile) {
  for (int w = 0; w < kImSize; w += w_tile) {
    for (int i = 0; i < kNum; i+=2) { // Which of the 256 filters we're on

        // LOOP 1: Set bias for each channel
          for(int hh = 0; hh < h_tile; hh++) {
            for(int ww = 0; ww < w_tile; ww++) {
              c_access(hh,ww) = bias[i];
              d_access(hh,ww) = bias[i+1];
            }
          }


        // LOOP 2: Convolution
          for (int j = 0; j < kNum; ++j) { // Which of the 256 channels we're on
            for (int p = 0; p < kKernel; ++p) { // Which row of the window/filter we're on
              for (int q = 0; q < kKernel; ++q) { // Which column of the window/filter we're on
                float weight_c = weight_access(i,j,p,q);
                float weight_d = weight_access(i+1,j,p,q);
                for(int hh = 0; hh < h_tile; hh++) {
                  for(int ww = 0; ww < w_tile; ww+=16) {
                    float16 input_vec = vload16(0, &input_access(j,(h + hh + p),(w + ww + q))), 
                                c_vec = vload16(0, &c_access(hh,ww)), 
                                d_vec = vload16(0, &d_access(hh,ww));
                    vstore16((c_vec + (input_vec * weight_c)), 0, &c_access(hh,ww));
                    vstore16((d_vec + (input_vec * weight_d)), 0, &d_access(hh,ww));
                  }
                }
              }
            }
          }


        // LOOP 1 & 3: Add bias and do ReLU
          for(int hh = 0; hh < h_tile; hh++) {
            for(int ww = 0; ww < w_tile; ww++) {
              c_access(hh,ww) = max(0.f, c_access(hh,ww));
              d_access(hh,ww) = max(0.f, d_access(hh,ww));
            }
          }


        // LOOP 4: Max pooling
          for(int hh = 0; hh < h_tile/2; hh++) {
            for(int ww = 0; ww < w_tile/2; ww++) {
              output_access(i,(h / 2 + hh),(w / 2 + ww)) = max(
                max(c_access((hh * 2),(ww * 2    )), c_access((hh * 2 + 1),(ww * 2    ))), 
                max(c_access((hh * 2),(ww * 2 + 1)), c_access((hh * 2 + 1),(ww * 2 + 1))));
              output_access(i+1,(h / 2 + hh),(w / 2 + ww)) = max(
                max(d_access((hh * 2),(ww * 2    )), d_access((hh * 2 + 1),(ww * 2    ))), 
                max(d_access((hh * 2),(ww * 2 + 1)), d_access((hh * 2 + 1),(ww * 2 + 1))));
            }
          }
          

      }
    }
  }
}