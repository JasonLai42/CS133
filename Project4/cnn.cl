#define kNum 256
#define kKernel 5
#define kImSize 224
#define kInImSize 228
#define kOutImSize 112

// Tile sizes to assign to work groups
#define i_tile 4
#define h_tile 16
#define w_tile 32

// Subtile sizes to assign to work items
#define h_subtile 4
#define w_subtile 32

// Arrays accessors
#define c_access(x,y,z) C[(x)*w_tile*w_tile+(y)*w_tile+(z)]
#define weight_access(w,x,y,z) weight[(w)*kNum*kKernel*kKernel+(x)*kKernel*kKernel+(y)*kKernel+(z)]
#define input_access(x,y,z) input[(x)*kInImSize*kInImSize+(y)*kInImSize+(z)]
#define output_access(x,y,z) output[(x)*kOutImSize*kOutImSize+(y)*kOutImSize+(z)]

__kernel
void CnnKernel(__constant float* input, __constant float* weight,
               __constant float* bias, __global float* output) {
  // your code goes here

  // Get work group and work item indices
  const int gGR = get_group_id(0), gGC = get_group_id(1);
  const int lR = get_local_id(0), lC = get_local_id(1);

  // Starting indices for rows/cols for the tiles assigned to each work group
  const int group_row_index = gGR * h_tile;
  const int group_col_index = gGC * w_tile;

  // Starting indices for the rows/cols for the subtiles assigned to each work item in each tile assigned to each work group
  const int item_row_index = lR * h_subtile;
  const int item_col_index = lC * w_subtile;

  // Intermediate image array to each handle a channel when unrolling i (same set of pixels, just different channels)
  __private float c_access(i_tile,h_tile,w_tile);

  for (int i = 0; i < kNum; i+=i_tile) { // Which of the 256 filters we're on

    // LOOP 1: Set bias for each channel
    for(int hh = 0; hh < h_subtile; hh++) {
      for(int ww = 0; ww < w_subtile; ww++) {
        c_access(0,hh,ww) = bias[i];
        c_access(1,hh,ww) = bias[i+1];
        c_access(2,hh,ww) = bias[i+2];
        c_access(3,hh,ww) = bias[i+3];
      }
    }


    // LOOP 2: Convolution
    for (int j = 0; j < kNum; ++j) { // Which of the 256 channels we're on
      for (int q = 0; q < kKernel; ++q) { // Which column of the window/filter we're on
        for (int p = 0; p < kKernel; ++p) { // Which row of the window/filter we're on
          float weight_c = weight_access(i,j,p,q), 
                weight_d = weight_access(i+1,j,p,q), 
                weight_e = weight_access(i+2,j,p,q), 
                weight_f = weight_access(i+3,j,p,q);
          for(int hh = 0; hh < h_subtile; hh++) {
            for(int ww = 0; ww < w_subtile; ww+=16) {
              float16 input_vec = vload16(0, &input_access(j,((group_row_index + item_row_index) + hh + p),((group_col_index + item_col_index) + ww + q))), 
                          c_vec = vload16(0, &c_access(0,hh,ww)), 
                          d_vec = vload16(0, &c_access(1,hh,ww)),
                          e_vec = vload16(0, &c_access(2,hh,ww)),
                          f_vec = vload16(0, &c_access(3,hh,ww));
              vstore16((c_vec + (input_vec * weight_c)), 0, &c_access(0,hh,ww));
              vstore16((d_vec + (input_vec * weight_d)), 0, &c_access(1,hh,ww));
              vstore16((e_vec + (input_vec * weight_e)), 0, &c_access(2,hh,ww));
              vstore16((f_vec + (input_vec * weight_f)), 0, &c_access(3,hh,ww));
            }
          }
        }
      }
    }


    // LOOP 1 & 3: Add bias and do ReLU
    for(int hh = 0; hh < h_subtile; hh++) {
      for(int ww = 0; ww < w_subtile; ww++) {
        c_access(0,hh,ww) = max(0.f, c_access(0,hh,ww));
        c_access(1,hh,ww) = max(0.f, c_access(1,hh,ww));
        c_access(2,hh,ww) = max(0.f, c_access(2,hh,ww));
        c_access(3,hh,ww) = max(0.f, c_access(3,hh,ww));
      }
    }


    // LOOP 4: Max pooling
    for(int hh = 0; hh < h_subtile/2; hh++) {
      for(int ww = 0; ww < w_subtile/2; ww++) {
        output_access(i,((group_row_index + item_row_index) / 2 + hh),((group_col_index + item_col_index) / 2 + ww)) = max(
          max(c_access(0,(hh * 2),(ww * 2    )), c_access(0,(hh * 2 + 1),(ww * 2    ))), 
          max(c_access(0,(hh * 2),(ww * 2 + 1)), c_access(0,(hh * 2 + 1),(ww * 2 + 1))));
        output_access(i+1,((group_row_index + item_row_index) / 2 + hh),((group_col_index + item_col_index) / 2 + ww)) = max(
          max(c_access(1,(hh * 2),(ww * 2    )), c_access(1,(hh * 2 + 1),(ww * 2    ))), 
          max(c_access(1,(hh * 2),(ww * 2 + 1)), c_access(1,(hh * 2 + 1),(ww * 2 + 1))));
        output_access(i+2,((group_row_index + item_row_index) / 2 + hh),((group_col_index + item_col_index) / 2 + ww)) = max(
          max(c_access(2,(hh * 2),(ww * 2    )), c_access(2,(hh * 2 + 1),(ww * 2    ))), 
          max(c_access(2,(hh * 2),(ww * 2 + 1)), c_access(2,(hh * 2 + 1),(ww * 2 + 1))));
        output_access(i+3,((group_row_index + item_row_index) / 2 + hh),((group_col_index + item_col_index) / 2 + ww)) = max(
          max(c_access(3,(hh * 2),(ww * 2    )), c_access(3,(hh * 2 + 1),(ww * 2    ))), 
          max(c_access(3,(hh * 2),(ww * 2 + 1)), c_access(3,(hh * 2 + 1),(ww * 2 + 1))));
      }
    }
    
  }
}
