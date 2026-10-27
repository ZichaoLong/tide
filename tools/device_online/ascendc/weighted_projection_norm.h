#pragma once
// Shared architecture N_d first-order contraction; actual vendor operands saved.
#include "weighted_allocation_math.h"
namespace tide_weighted_projection_device {
using namespace tide_weighted_device;
__aicore__ inline void reverse_norm(__gm__ float* normal,__gm__ float* rstd,
    __gm__ float* gradient,__gm__ uint8_t* roots,__gm__ int32_t* error,I rows,I width){
  if(error[0])return;
  for(I r=0;r<rows;++r){if(!roots[r]){for(I d=0;d<width;++d)gradient[r*width+d]=0;continue;}
    float dot=0;for(I d=0;d<width;++d)dot+=normal[r*width+d]*gradient[r*width+d];dot/=float(width);
    for(I d=0;d<width;++d){auto g=gradient+r*width+d;*g=(*g-normal[r*width+d]*dot)*rstd[r];
      if(!finite(*g)){error[0]=23;return;}}
  }
}
}
