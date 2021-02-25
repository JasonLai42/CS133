#define kNum 256
#define kKernel 5
#define kImSize 224
#define kInImSize 228
#define kOutImSize 112

// Tile sizes to assign to work groups
#define i_tile 4
#define h_tile 4
#define w_tile 4

// Subtile sizes to assign to work items
#define h_subtile 4
#define w_subtile 32

// Arrays accessors
#define c_init(x,y,z) C[(x)*w_tile*w_tile+(y)*w_tile+(z)] __attribute__((aligned(32 * sizeof(float))))
#define c_access(x,y,z) C[(x)*w_tile*w_tile+(y)*w_tile+(z)]
#define weight_access(w,x,y,z) weight[(w)*kNum*kKernel*kKernel+(x)*kKernel*kKernel+(y)*kKernel+(z)]
#define input_access(x,y,z) input[(x)*kInImSize*kInImSize+(y)*kInImSize+(z)]
#define output_access(x,y,z) output[(x)*kOutImSize*kOutImSize+(y)*kOutImSize+(z)]

__kernel
void CnnKernel(__constant float* input, __constant float* weight,
               __constant float* bias, __global float* output) {
  // your code goes here

  // Get work group and work item indices
  const int gGCH = get_group_id(0), gGR = get_group_id(1), gGC = get_group_id(2);
  const int lCH = get_local_id(0), lR = get_local_id(1), lC = get_local_id(2);

  // Starting indices for rows/cols for the tiles assigned to each work group
  const int group_channel_index = gGR;
  const int group_row_index = gGR * h_tile;
  const int group_col_index = gGC * w_tile;

  // Starting indices for the rows/cols for the subtiles assigned to each work item in each tile assigned to each work group
  const int item_channel_index = lCH;
  const int item_row_index = lR * h_subtile;
  const int item_col_index = lC * w_subtile;


    // Intermediate image array to each handle a channel when unrolling i (same set of pixels, just different channels)
    __private float c_init(i_tile,h_tile,w_tile) = {{{ 0 }}};

    // LOOP 2: Convolution
    for (int j = 0; j < kNum; ++j) { // Which of the 256 channels we're on
      for (int q = 0; q < kKernel; ++q) { // Which column of the window/filter we're on
        for (int p = 0; p < kKernel; ++p) { // Which row of the window/filter we're on
          float weight_c = weight_access(group_channel_index,j,p,q);
          for(int hh = 0; hh < h_tile; hh++) {
            for(int ww = 0; ww < w_tile; ww++) {
              c_access(0,hh,ww) += weight_c * input_access(j,((group_row_index) + hh + p),((group_col_index) + ww + q));
              c_access(1,hh,ww) += weight_d * input_access(j,((group_row_index) + hh + p),((group_col_index) + ww + q));
              c_access(2,hh,ww) += weight_e * input_access(j,((group_row_index) + hh + p),((group_col_index) + ww + q));
              c_access(3,hh,ww) += weight_f * input_access(j,((group_row_index) + hh + p),((group_col_index) + ww + q));
            }
          }
        }
      }
    }


    // LOOP 1 & 3: Add bias and do ReLU
    for(int hh = 0; hh < h_tile; hh++) {
      for(int ww = 0; ww < w_tile; ww++) {
        c_access(0,hh,ww) = max(0.f, c_access(0,hh,ww) + bias[group_channel_index]);
      }
    }


    // LOOP 4: Max pooling
        output_access(group_channel_index,((group_row_index + item_row_index) / 2 + hh),((group_col_index + item_col_index) / 2 + ww)) = max(
          max(c_access(0,0,0), c_access(0,1,0)), 
          max(c_access(0,0,1), c_access(0,1,1)));
}
