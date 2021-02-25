#define kNum 256
#define kKernel 5
#define kImSize 224
#define kInImSize 228
#define kOutImSize 112

// Tile sizes to assign to work groups
#define i_tile 32
#define h_tile 8
#define w_tile 32

// Subtile sizes to assign to work items
#define i_subtile 8
#define h_subtile 2
#define w_subtile 8

// Arrays accessors
// #define c_init(y,z) C[(y)*w_tile+(z)] __attribute__((aligned(32 * sizeof(float))))
// #define c_access(y,z) C[(y)*w_tile+(z)]
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
  const int group_channel_index = gGCH * i_tile;
  const int group_row_index = gGR * h_tile;
  const int group_col_index = gGC * w_tile;

  // Starting indices for rows/cols for the subtiles assigned to each work group
  const int item_channel_index = group_channel_index + (lCH * i_subtile);
  const int item_row_index = group_row_index + (lR * h_subtile);
  const int item_col_index = group_col_index + (lC * w_subtile);


  // for (int i = 0; i < kNum; i+=i_tile) { // Which of the 256 filters we're on

    // Intermediate image array to each handle a channel when unrolling i (same set of pixels, just different channels)
    __private float c_init(i_subtile,h_subtile,w_subtile) = {{{ 0 }}};

    // LOOP 2: Convolution
    for (int j = 0; j < kNum; ++j) { // Which of the 256 channels we're on
      for (int q = 0; q < kKernel; ++q) { // Which column of the window/filter we're on
        for (int p = 0; p < kKernel; ++p) { // Which row of the window/filter we're on
          float weight_c = weight_access(item_channel_index,j,p,q), 
                weight_d = weight_access(item_channel_index+1,j,p,q), 
                weight_e = weight_access(item_channel_index+2,j,p,q), 
                weight_f = weight_access(item_channel_index+3,j,p,q),
                weight_g = weight_access(item_channel_index+4,j,p,q), 
                weight_h = weight_access(item_channel_index+5,j,p,q), 
                weight_i = weight_access(item_channel_index+6,j,p,q), 
                weight_j = weight_access(item_channel_index+7,j,p,q);
          for(int hh = 0; hh < h_subtile; hh++) {
            for(int ww = 0; ww < w_subtile; ww++) {
              c_access(0,hh,ww) += weight_c * input_access(j,(item_row_index + hh + p),(item_col_index + ww + q));
              c_access(1,hh,ww) += weight_d * input_access(j,(item_row_index + hh + p),(item_col_index + ww + q));
              c_access(2,hh,ww) += weight_e * input_access(j,(item_row_index + hh + p),(item_col_index + ww + q));
              c_access(3,hh,ww) += weight_f * input_access(j,(item_row_index + hh + p),(item_col_index + ww + q));
              c_access(4,hh,ww) += weight_g * input_access(j,(item_row_index + hh + p),(item_col_index + ww + q));
              c_access(5,hh,ww) += weight_h * input_access(j,(item_row_index + hh + p),(item_col_index + ww + q));
              c_access(6,hh,ww) += weight_i * input_access(j,(item_row_index + hh + p),(item_col_index + ww + q));
              c_access(7,hh,ww) += weight_j * input_access(j,(item_row_index + hh + p),(item_col_index + ww + q));
            }
          }
        }
      }
    }


    // LOOP 1 & 3: Add bias and do ReLU
    for(int hh = 0; hh < h_subtile; hh++) {
      for(int ww = 0; ww < w_subtile; ww++) {
        c_access(0,hh,ww) = max(0.f, c_access(0,hh,ww) + bias[item_channel_index]);
        c_access(1,hh,ww) = max(0.f, c_access(1,hh,ww) + bias[item_channel_index+1]);
        c_access(2,hh,ww) = max(0.f, c_access(2,hh,ww) + bias[item_channel_index+2]);
        c_access(3,hh,ww) = max(0.f, c_access(3,hh,ww) + bias[item_channel_index+3]);
        c_access(4,hh,ww) = max(0.f, c_access(4,hh,ww) + bias[item_channel_index+4]);
        c_access(5,hh,ww) = max(0.f, c_access(5,hh,ww) + bias[item_channel_index+5]);
        c_access(6,hh,ww) = max(0.f, c_access(6,hh,ww) + bias[item_channel_index+6]);
        c_access(7,hh,ww) = max(0.f, c_access(7,hh,ww) + bias[item_channel_index+7]);
      }
    }


    // LOOP 4: Max pooling
    for(int hh = 0; hh < h_subtile/2; hh++) {
      for(int ww = 0; ww < w_subtile/2; ww++) {
        output_access(item_channel_index,(item_row_index / 2 + hh),(item_col_index / 2 + ww)) = max(
          max(c_access(0,(hh * 2),(ww * 2    )), c_access(0,(hh * 2 + 1),(ww * 2    ))), 
          max(c_access(0,(hh * 2),(ww * 2 + 1)), c_access(0,(hh * 2 + 1),(ww * 2 + 1))));
        output_access(item_channel_index+1,(item_row_index / 2 + hh),(item_col_index / 2 + ww)) = max(
          max(c_access(1,(hh * 2),(ww * 2    )), c_access(1,(hh * 2 + 1),(ww * 2    ))), 
          max(c_access(1,(hh * 2),(ww * 2 + 1)), c_access(1,(hh * 2 + 1),(ww * 2 + 1))));
        output_access(item_channel_index+2,(item_row_index / 2 + hh),(item_col_index / 2 + ww)) = max(
          max(c_access(2,(hh * 2),(ww * 2    )), c_access(2,(hh * 2 + 1),(ww * 2    ))), 
          max(c_access(2,(hh * 2),(ww * 2 + 1)), c_access(2,(hh * 2 + 1),(ww * 2 + 1))));
        output_access(item_channel_index+3,(item_row_index / 2 + hh),(item_col_index / 2 + ww)) = max(
          max(c_access(3,(hh * 2),(ww * 2    )), c_access(3,(hh * 2 + 1),(ww * 2    ))), 
          max(c_access(3,(hh * 2),(ww * 2 + 1)), c_access(3,(hh * 2 + 1),(ww * 2 + 1))));
        output_access(item_channel_index+4,(item_row_index / 2 + hh),(item_col_index / 2 + ww)) = max(
          max(c_access(4,(hh * 2),(ww * 2    )), c_access(4,(hh * 2 + 1),(ww * 2    ))), 
          max(c_access(4,(hh * 2),(ww * 2 + 1)), c_access(4,(hh * 2 + 1),(ww * 2 + 1))));
        output_access(item_channel_index+5,(item_row_index / 2 + hh),(item_col_index / 2 + ww)) = max(
          max(c_access(5,(hh * 2),(ww * 2    )), c_access(5,(hh * 2 + 1),(ww * 2    ))), 
          max(c_access(5,(hh * 2),(ww * 2 + 1)), c_access(5,(hh * 2 + 1),(ww * 2 + 1))));
        output_access(item_channel_index+6,(item_row_index / 2 + hh),(item_col_index / 2 + ww)) = max(
          max(c_access(6,(hh * 2),(ww * 2    )), c_access(6,(hh * 2 + 1),(ww * 2    ))), 
          max(c_access(6,(hh * 2),(ww * 2 + 1)), c_access(6,(hh * 2 + 1),(ww * 2 + 1))));
        output_access(item_channel_index+7,(item_row_index / 2 + hh),(item_col_index / 2 + ww)) = max(
          max(c_access(7,(hh * 2),(ww * 2    )), c_access(7,(hh * 2 + 1),(ww * 2    ))), 
          max(c_access(7,(hh * 2),(ww * 2 + 1)), c_access(7,(hh * 2 + 1),(ww * 2 + 1))));
      }
    }
    
  // }
}
