#pragma once
#include "weighted_allocation_math.h"
namespace tide_weighted_residual_device {
using namespace tide_weighted_device;
__aicore__ inline void guard(__gm__ I* owners,__gm__ float* input,__gm__ uint8_t* roots,
    __gm__ float* output,__gm__ uint8_t* connected,__gm__ int32_t* error,I rows,I width,I reverse){
  if(error[0])return;for(I r=0;r<rows;++r){const bool active=owners[r]>=0&&(!reverse||roots[r]);
    if(reverse)connected[r]=active;
    for(I j=0;j<width;++j){float x=0;if(active){x=input[r*width+j];if(!finite(x)){error[0]=35;return;}}output[r*width+j]=x;}
  }
}
}
