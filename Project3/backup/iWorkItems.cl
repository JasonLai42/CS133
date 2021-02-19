__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;

#define c_access(y,z) C[(y)*kImSize+(z)]
#define weight_access(w,x,y,z) weight[(w)*kNum*kKernel*kKernel+(x)*kKernel*kKernel+(y)*kKernel+(z)]
#define input_access(x,y,z) input[(x)*kInImSize*kInImSize+(y)*kInImSize+(z)]
#define output_access(x,y,z) output[(x)*kOutImSize*kOutImSize+(y)*kOutImSize+(z)]

__kernel
void CnnKernel(__constant float* input, __constant float* weight,
               __constant float* bias, __global float* output) {
  // your code goes here

  // Tile size to prevent out of resource
  const int i_tile = 8;
  const int h_tile = 2;
  const int w_tile = 32;

  // Get work group and work item indices
  const int gGCH = get_group_id(0), gGR = get_group_id(1), gGC = get_group_id(2);
  const int lCH = get_local_id(0), lR = get_local_id(1), lC = get_local_id(2);

  // Starting indices for rows/cols of result for the tiles assigned to each work group
  const group_channel_index = gGCH * i_tile;
  const group_row_index = gGR * h_tile;
  const group_col_index = gGC * w_tile;

  // Intermediate image array
  __private float c_access(h_tile,w_tile) = { 0 };



//for (int i = 0; i < kNum; i+=i_tile) { // Which of the 256 filters we're on

        // LOOP 2: Convolution
          for (int j = 0; j < kNum; ++j) { // Which of the 256 channels we're on
            for (int p = 0; p < kKernel; ++p) { // Which row of the window/filter we're on
              for (int q = 0; q < kKernel; ++q) { // Which column of the window/filter we're on
                float weight_c = weight_access((group_channel_index + lCH),j,p,q);
                for(int hh = 0; hh < h_tile; hh++) {
                  for(int ww = 0; ww < w_tile; ww+=16) {
                    float16 input_vec = vload16(0, &input_access(j,(group_row_index + hh + p),(group_col_index + ww + q))), 
                                c_vec = vload16(0, &c_access(hh,ww));
                    vstore16((c_vec + (input_vec * weight_c)), 0, &c_access(hh,ww));
                  }
                }
              }
            }
          }


        // LOOP 1 & 3: Add bias and do ReLU
          for(int hh = 0; hh < h_tile; hh++) {
            for(int ww = 0; ww < w_tile; ww++) {
              c_access(hh,ww) = max(0.f, (c_access(hh,ww) + bias[group_channel_index + lCH]));
            }
          }


        // LOOP 4: Max pooling
          for(int hh = 0; hh < h_tile/2; hh++) {
            for(int ww = 0; ww < w_tile/2; ww++) {
              output_access((group_channel_index + lCH),(group_row_index / 2 + hh),(group_col_index / 2 + ww)) = max(
                max(c_access((hh * 2),(ww * 2    )), c_access((hh * 2 + 1),(ww * 2    ))), 
                max(c_access((hh * 2),(ww * 2 + 1)), c_access((hh * 2 + 1),(ww * 2 + 1))));
            }
          }
          

  //}
}
