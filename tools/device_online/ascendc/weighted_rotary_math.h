#pragma once
#include "weighted_numeric_time.h"
namespace tide_weighted_rotary_device {
using namespace tide_weighted_device;
__aicore__ inline void phase(__gm__ I* ages,__gm__ float* frequencies,__gm__ float* output,
    __gm__ int32_t* error,I count,I pairs,float scale){
  if(error[0])return;for(I i=0;i<count;++i){if(ages[i]<0){error[0]=33;return;}
    const float displacement=-tide_weighted_time_device::as_float(ages[i])/scale;
    for(I j=0;j<pairs;++j){const float x=displacement*frequencies[j];if(!finite(x)){error[0]=33;return;}output[i*pairs+j]=x;}}
}
__aicore__ inline void apply(__gm__ float* input,__gm__ float* cosine,__gm__ float* sine,
    __gm__ uint8_t* roots,__gm__ float* output,__gm__ int32_t* error,I rows,I capacity,I width,I pairs,I reverse){
  if(error[0])return;
  for(I r=0;r<rows;++r)for(I i=0;i<capacity;++i){const I a=r*capacity+i;
    if(reverse&&!roots[r]){for(I j=0;j<width;++j)output[a*width+j]=0;continue;}
    for(I j=0;j<width-2*pairs;++j){const float x=input[a*width+j];if(!finite(x)){error[0]=reverse?34:33;return;}output[a*width+j]=x;}
    for(I j=0;j<pairs;++j){const I off=a*width+width-2*pairs+2*j;const float c=cosine[a*pairs+j],s=reverse?-sine[a*pairs+j]:sine[a*pairs+j];
      const float x=input[off],y=input[off+1],u=x*c-y*s,v=x*s+y*c;
      if(!finite(x)||!finite(y)||!finite(c)||!finite(s)||!finite(u)||!finite(v)){error[0]=reverse?34:33;return;}
      output[off]=u;output[off+1]=v;
    }
  }
}
}
