#pragma once
#include "weighted_allocation_math.h"
namespace tide_weighted_kv_device {
using namespace tide_weighted_device;
__aicore__ inline void prepare(__gm__ float* h,__gm__ I* owners,__gm__ uint8_t* present,
    __gm__ float* u,__gm__ float* mass,__gm__ I* selected,__gm__ I* indices,__gm__ I* row_ids,
    __gm__ uint8_t* positive,__gm__ int32_t* error,I rows,I width,I banks){
  if(error[0])return;
  for(I r=0;r<rows;++r){mass[r]=0;positive[r]=0;selected[r]=-1;indices[r]=0;row_ids[r]=-1;
    for(I d=0;d<width;++d)u[r*width+d]=0;
    if(!present[r])continue;const float m=h[r*(width+1)];if(!finite(m)){error[0]=20;return;}mass[r]=m;
    if(m<=0)continue;if(owners[r]<0||owners[r]>=banks){error[0]=2;return;}
    positive[r]=1;selected[r]=owners[r];indices[r]=owners[r];row_ids[r]=r;
    for(I d=0;d<width;++d){const float x=h[r*(width+1)+d+1];if(!finite(x)){error[0]=20;return;}u[r*width+d]=x;}
  }
}
__aicore__ inline void roots(__gm__ uint8_t* raw,__gm__ uint8_t* kv,__gm__ uint8_t* summary,__gm__ int32_t* error,I rows){
  if(error[0])return;for(I r=0;r<rows;++r){kv[r]=raw[r*5+1]||raw[r*5+2];summary[r]=raw[r*5+3];}
}
__aicore__ inline void merge(__gm__ float* mass,__gm__ uint8_t* mass_roots,__gm__ float* kv,
    __gm__ uint8_t* kv_roots,__gm__ float* summary,__gm__ uint8_t* summary_roots,
    __gm__ float* output,__gm__ uint8_t* connected,__gm__ int32_t* error,I rows,I width){
  if(error[0])return;
  for(I r=0;r<rows;++r){connected[r]=mass_roots[r]||kv_roots[r]||summary_roots[r];output[r*(width+1)]=mass_roots[r]?mass[r]:0;
    if(!finite(output[r*(width+1)])){error[0]=23;return;}
    for(I d=0;d<width;++d){const float value=(kv_roots[r]?kv[r*width+d]:0)+(summary_roots[r]?summary[r*width+d]:0);
      output[r*(width+1)+d+1]=value;if(!finite(value)){error[0]=23;return;}}
  }
}
}
