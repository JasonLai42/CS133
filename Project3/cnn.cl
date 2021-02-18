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

  const int i_tile = 16;
  const int h_tile = 16;
  const int w_tile = 16;

  const int h_subtile = 4;
  const int w_subtile = 4;

  const int gGR = get_group_id(0), gGC = get_group_id(1);
  const int lR = get_local_id(0), lC = get_local_id(1);

  // Starting indices for rows/cols of result for the tiles assigned to each work group
  const group_row_index = gGR * h_tile;
  const group_col_index = gGC * w_tile;

  // Starting indices for row/cols of intermediate C array for each work item
  const item_row_index = lR * h_subtile;
  const item_col_index = lC * h_subtile;
  const item_row_bound = item_row_index + h_subtile;
  const item_col_bound = item_col_index + h_subtile;

  // Intermediate image array
  __local float c_access(h_tile,w_tile);

  for (int i = 0; i < kNum; ++i) { // Which of the 256 filters we're on
    // for (int h = 0; h < kImSize; h += h_tile) {
    //   for (int w = 0; w < kImSize; w += w_tile) {


        // LOOP 1: Convolution layer
          for(int hh = 0; hh < h_tile; hh++) {
            for(int ww = 0; ww < w_tile; ww++) {

              // Set bias for each channel
              c_access(hh,ww) = bias[i];

              // Convolution
              for (int j = 0; j < kNum; ++j) { // Which of the 256 channels we're on
                for (int p = 0; p < kKernel; ++p) { // Which row of the window/filter we're on
                  for (int q = 0; q < kKernel; ++q) // Which column of the window/filter we're on
                    c_access(hh,ww) += weight_access(i,j,p,q) * input_access(j,(group_row_index + hh + p),(group_col_index + ww + q));
                }
              }

              // ReLU
              c_access(hh,ww) = max(0.f, c_access(hh,ww));

            }
          }

        // barrier(CLK_LOCAL_MEM_FENCE);

        // LOOP 2: Max pooling layer
          for(int hh = 0; hh < h_tile/2; hh++) {
            for(int ww = 0; ww < w_tile/2; ww++) {
              output_access(i,(group_row_index / 2 + hh),(group_col_index / 2 + ww)) = max(
                max(c_access((hh * 2),(ww * 2    )), c_access((hh * 2 + 1),(ww * 2    ))), 
                max(c_access((hh * 2),(ww * 2 + 1)), c_access((hh * 2 + 1),(ww * 2 + 1))));
            }
          }
          

    //   }
    // }
  }
}
