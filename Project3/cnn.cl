__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;

#define c_access(y,z) C[(y)*kImSize+(z)]
#define d_access(y,z) D[(y)*kImSize+(z)]
#define e_access(y,z) E[(y)*kImSize+(z)]
#define f_access(y,z) F[(y)*kImSize+(z)]
#define c1_access(y,z) G[(y)*kImSize+(z)]
#define d1_access(y,z) H[(y)*kImSize+(z)]
#define e1_access(y,z) I[(y)*kImSize+(z)]
#define f1_access(y,z) J[(y)*kImSize+(z)]
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
  const int gGR = get_group_id(0), gGC = get_group_id(1);
  const int lR = get_local_id(0), lC = get_local_id(1);

  // Starting indices for rows/cols of result for the tiles assigned to each work group
  const group_row_index = gGR * h_tile;
  const group_col_index = gGC * w_tile;

  // Intermediate image array
  __local float c_access(h_tile,w_tile);
  __local float d_access(h_tile,w_tile);
  __local float e_access(h_tile,w_tile);
  __local float f_access(h_tile,w_tile);
  __local float c1_access(h_tile,w_tile);
  __local float d1_access(h_tile,w_tile);
  __local float e1_access(h_tile,w_tile);
  __local float f1_access(h_tile,w_tile);



for (int i = 0; i < kNum; i+=i_tile) { // Which of the 256 filters we're on

        // LOOP 1: Set bias for each channel
          for(int hh = 0; hh < h_tile; hh++) {
            for(int ww = 0; ww < w_tile; ww++) {
              c_access(hh,ww) = bias[i];
              d_access(hh,ww) = bias[i+1];
              e_access(hh,ww) = bias[i+2];
              f_access(hh,ww) = bias[i+3];
              c1_access(hh,ww) = bias[i+4];
              d1_access(hh,ww) = bias[i+5];
              e1_access(hh,ww) = bias[i+6];
              f1_access(hh,ww) = bias[i+7];
            }
          }


        // LOOP 2: Convolution
          for (int j = 0; j < kNum; ++j) { // Which of the 256 channels we're on
            for (int p = 0; p < kKernel; ++p) { // Which row of the window/filter we're on
              for (int q = 0; q < kKernel; ++q) { // Which column of the window/filter we're on
                float weight_c = weight_access(i,j,p,q), 
                      weight_d = weight_access(i+1,j,p,q), 
                      weight_e = weight_access(i+2,j,p,q), 
                      weight_f = weight_access(i+3,j,p,q), 
                      weight_c1 = weight_access(i+4,j,p,q), 
                      weight_d1 = weight_access(i+5,j,p,q), 
                      weight_e1 = weight_access(i+6,j,p,q), 
                      weight_f1 = weight_access(i+7,j,p,q);
                for(int hh = 0; hh < h_tile; hh++) {
                  for(int ww = 0; ww < w_tile; ww+=16) {
                    float16 input_vec = vload16(0, &input_access(j,(group_row_index + hh + p),(group_col_index + ww + q))), 
                                c_vec = vload16(0, &c_access(hh,ww)), 
                                d_vec = vload16(0, &d_access(hh,ww)),
                                e_vec = vload16(0, &e_access(hh,ww)),
                                f_vec = vload16(0, &f_access(hh,ww)),
                                c1_vec = vload16(0, &c1_access(hh,ww)), 
                                d1_vec = vload16(0, &d1_access(hh,ww)),
                                e1_vec = vload16(0, &e1_access(hh,ww)),
                                f1_vec = vload16(0, &f1_access(hh,ww));
                    vstore16((c_vec + (input_vec * weight_c)), 0, &c_access(hh,ww));
                    vstore16((d_vec + (input_vec * weight_d)), 0, &d_access(hh,ww));
                    vstore16((e_vec + (input_vec * weight_e)), 0, &e_access(hh,ww));
                    vstore16((f_vec + (input_vec * weight_f)), 0, &f_access(hh,ww));
                    vstore16((c1_vec + (input_vec * weight_c1)), 0, &c1_access(hh,ww));
                    vstore16((d1_vec + (input_vec * weight_d1)), 0, &d1_access(hh,ww));
                    vstore16((e1_vec + (input_vec * weight_e1)), 0, &e1_access(hh,ww));
                    vstore16((f1_vec + (input_vec * weight_f1)), 0, &f1_access(hh,ww));
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
              e_access(hh,ww) = max(0.f, e_access(hh,ww));
              f_access(hh,ww) = max(0.f, f_access(hh,ww));
              c1_access(hh,ww) = max(0.f, c1_access(hh,ww));
              d1_access(hh,ww) = max(0.f, d1_access(hh,ww));
              e1_access(hh,ww) = max(0.f, e1_access(hh,ww));
              f1_access(hh,ww) = max(0.f, f1_access(hh,ww));
            }
          }


        // LOOP 4: Max pooling
          for(int hh = 0; hh < h_tile/2; hh++) {
            for(int ww = 0; ww < w_tile/2; ww++) {
              output_access(i,(group_row_index / 2 + hh),(group_col_index / 2 + ww)) = max(
                max(c_access((hh * 2),(ww * 2    )), c_access((hh * 2 + 1),(ww * 2    ))), 
                max(c_access((hh * 2),(ww * 2 + 1)), c_access((hh * 2 + 1),(ww * 2 + 1))));
              output_access(i+1,(group_row_index / 2 + hh),(group_col_index / 2 + ww)) = max(
                max(d_access((hh * 2),(ww * 2    )), d_access((hh * 2 + 1),(ww * 2    ))), 
                max(d_access((hh * 2),(ww * 2 + 1)), d_access((hh * 2 + 1),(ww * 2 + 1))));
              output_access(i+2,(group_row_index / 2 + hh),(group_col_index / 2 + ww)) = max(
                max(e_access((hh * 2),(ww * 2    )), e_access((hh * 2 + 1),(ww * 2    ))), 
                max(e_access((hh * 2),(ww * 2 + 1)), e_access((hh * 2 + 1),(ww * 2 + 1))));
              output_access(i+3,(group_row_index / 2 + hh),(group_col_index / 2 + ww)) = max(
                max(f_access((hh * 2),(ww * 2    )), f_access((hh * 2 + 1),(ww * 2    ))), 
                max(f_access((hh * 2),(ww * 2 + 1)), f_access((hh * 2 + 1),(ww * 2 + 1))));
              output_access(i+4,(group_row_index / 2 + hh),(group_col_index / 2 + ww)) = max(
                max(c1_access((hh * 2),(ww * 2    )), c1_access((hh * 2 + 1),(ww * 2    ))), 
                max(c1_access((hh * 2),(ww * 2 + 1)), c1_access((hh * 2 + 1),(ww * 2 + 1))));
              output_access(i+5,(group_row_index / 2 + hh),(group_col_index / 2 + ww)) = max(
                max(d1_access((hh * 2),(ww * 2    )), d1_access((hh * 2 + 1),(ww * 2    ))), 
                max(d1_access((hh * 2),(ww * 2 + 1)), d1_access((hh * 2 + 1),(ww * 2 + 1))));
              output_access(i+6,(group_row_index / 2 + hh),(group_col_index / 2 + ww)) = max(
                max(e1_access((hh * 2),(ww * 2    )), e1_access((hh * 2 + 1),(ww * 2    ))), 
                max(e1_access((hh * 2),(ww * 2 + 1)), e1_access((hh * 2 + 1),(ww * 2 + 1))));
              output_access(i+7,(group_row_index / 2 + hh),(group_col_index / 2 + ww)) = max(
                max(f1_access((hh * 2),(ww * 2    )), f1_access((hh * 2 + 1),(ww * 2    ))), 
                max(f1_access((hh * 2),(ww * 2 + 1)), f1_access((hh * 2 + 1),(ww * 2 + 1))));
            }
          }
          

  }
}
