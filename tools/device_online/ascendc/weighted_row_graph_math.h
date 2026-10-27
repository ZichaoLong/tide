#pragma once
#include "weighted_allocation_math.h"
namespace tide_weighted_row_device {
using namespace tide_weighted_device;
__aicore__ inline void bind(__gm__ I* owners,__gm__ float* input,__gm__ float* output,__gm__ int32_t* error,
    I rows,I width,I banks,I positive){
  if(error[0])return;for(I r=0;r<rows;++r){if(owners[r]< -1||owners[r]>=banks){error[0]=2;return;}
    for(I j=0;j<width;++j){float x=0;if(owners[r]>=0){x=input[r*width+j];if(!finite(x)||(positive&&x<0)){error[0]=36;return;}}output[r*width+j]=x;}}
}
__aicore__ inline void accumulate(__gm__ I* owners,__gm__ float* input,__gm__ uint8_t* roots,
    __gm__ float* output,__gm__ uint8_t* connected,__gm__ int32_t* error,I rows,I width){
  if(error[0])return;for(I r=0;r<rows;++r){if(owners[r]<0||!roots[r])continue;connected[r]=1;
    for(I j=0;j<width;++j){const I i=r*width+j;const float x=input[i],sum=output[i]+x;
      if(!finite(x)||!finite(sum)){error[0]=37;return;}output[i]=sum;}}
}
__aicore__ inline void derivative(__gm__ float* input,__gm__ float* value,__gm__ float* gradient,
    __gm__ uint8_t* roots,__gm__ float* output,__gm__ int32_t* error,I rows,I width,I mode,float scalar){
  if(error[0])return;Math math;math.init();for(I r=0;r<rows;++r)for(I j=0;j<width;++j){const I i=r*width+j;float dx=0;
    if(roots[r]){const float x=input[i],y=value[i],g=gradient[i];
      if(mode==0)dx=g*y*(1-y);else if(mode==1)dx=g/(1+x);else if(mode==2)dx=x<=scalar?g:0;
      else if(mode==3)dx=g*y;else if(mode==4)dx=g/(2*y);
      else dx=g*(x>20?1:math.sigmoid(x));
      if(!finite(dx)){error[0]=37;return;}}
    output[i]=dx;}
}
}
