#define kNum 256
#define kKernel 5
#define kImSize 224
#define kInImSize 228
#define kOutImSize 112

// Tile sizes to assign to work groups
#define i_tile 32   // should be equal to (LOCAL_SIZE[0] * i_subtile)
#define h_tile 8    // should be equal to (LOCAL_SIZE[1] * h_subtile)
#define w_tile 32   // should be equal to (LOCAL_SIZE[2] * w_subtile)

// Subtile sizes to assign to work items
#define i_subtile 8 // should be equal to (256 / GLOBAL_SIZE[0])
#define h_subtile 2 // should be equal to (224 / GLOBAL_SIZE[1])
#define w_subtile 8 // should be equal to (224 / GLOBAL_SIZE[2])

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
  const int group_channel_index = gGCH * i_tile;
  const int group_row_index = gGR * h_tile;
  const int group_col_index = gGC * w_tile;

  // Starting indices for rows/cols for the subtiles assigned to each work group
  const int item_channel_index = group_channel_index + (lCH * i_subtile);
  const int item_row_index = group_row_index + (lR * h_subtile);
  const int item_col_index = group_col_index + (lC * w_subtile);

  // Intermediate image array to each handle a channel when unrolling i (same set of pixels, just different channels)
  __private float c_init(i_subtile,h_subtile,w_subtile) = {{{ 0 }}};

  // Array to hold weights (filters) for unrolling
  __local float curr_weights[i_subtile];

  // LOOP 2: Convolution
  for (int j = 0; j < kNum; ++j) { // Which of the 256 channels we're on
    for (int q = 0; q < kKernel; ++q) { // Which column of the window/filter we're on
      for (int p = 0; p < kKernel; ++p) { // Which row of the window/filter we're on
        #pragma unroll i_subtile
        for(int k = 0; k < i_subtile; k++) {
          curr_weights[k] = weight_access(item_channel_index + k,j,p,q);
        }
        for(int hh = 0; hh < h_subtile; hh++) {
          for(int ww = 0; ww < w_subtile; ww++) {
            #pragma unroll i_subtile
            for(int i = 0; i < i_subtile; i++) {
              c_access(i,hh,ww) += curr_weights[i] * input_access(j,(item_row_index + hh + p),(item_col_index + ww + q));
            }
          }
        }
      }
    }
  }


  // LOOP 1 & 3: Add bias and do ReLU
  for(int hh = 0; hh < h_subtile; hh++) {
    for(int ww = 0; ww < w_subtile; ww++) {
      #pragma unroll i_subtile
      for(int i = 0; i < i_subtile; i++) {
        c_access(i,hh,ww) = max(0.f, c_access(i,hh,ww) + bias[item_channel_index + i]);
      }
    }
  }


  // LOOP 4: Max pooling
  for(int hh = 0; hh < h_subtile/2; hh++) {
    for(int ww = 0; ww < w_subtile/2; ww++) {
      #pragma unroll i_subtile
      for(int i = 0; i < i_subtile; i++) {
        output_access((item_channel_index + i),(item_row_index / 2 + hh),(item_col_index / 2 + ww)) = max(
          max(c_access(i,(hh * 2),(ww * 2    )), c_access(i,(hh * 2 + 1),(ww * 2    ))), 
          max(c_access(i,(hh * 2),(ww * 2 + 1)), c_access(i,(hh * 2 + 1),(ww * 2 + 1))));
      }
    }
  }
    
}
