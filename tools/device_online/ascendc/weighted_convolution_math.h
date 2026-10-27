#pragma once
// Fixed-time convolution §2: sparse original times, current unweighted feature.
#include "weighted_allocation_math.h"
namespace tide_weighted_conv_device {
using namespace tide_weighted_device;
__aicore__ inline void prepare(__gm__ float* old,__gm__ I* meta,__gm__ I* time,
    __gm__ float* current,__gm__ I* owners,__gm__ I* indices,__gm__ I* mapping,
    __gm__ float* taps,__gm__ int32_t* error,I rows,I raw_width,I capacity,I offset,I features,I size,I stride,I banks){
  if(error[0])return;const I iw=4+capacity;
  for(I r=0;r<rows;++r){indices[r]=0;for(I k=0;k<size;++k){mapping[r*size+k]=-1;for(I d=0;d<features;++d)taps[(r*size+k)*features+d]=0;}
    if(owners[r]==-1)continue;if(owners[r]<0||owners[r]>=banks||time[r]<0){error[0]=27;return;}
    const I n=meta[r*iw+2];if(n<0||n>capacity||n>(size-1)*stride+1){error[0]=27;return;}indices[r]=owners[r];
    for(I i=0;i<n;++i){const I stamp=meta[r*iw+4+i];
      if(stamp<0||(i&&stamp<=meta[r*iw+3+i])){error[0]=27;return;}
      for(I d=0;d<features;++d)if(!finite(old[r*raw_width+offset+i*features+d])){error[0]=27;return;}}
    for(I d=0;d<features;++d){const float x=current[r*features+d];if(!finite(x)){error[0]=27;return;}taps[r*size*features+d]=x;}
    for(I k=1;k<size;++k){const I delta=k*stride;if(time[r]<delta)continue;const I target=time[r]-delta;
      for(I i=0;i<n;++i)if(meta[r*iw+4+i]==target){mapping[r*size+k]=i;
        for(I d=0;d<features;++d)taps[(r*size+k)*features+d]=old[r*raw_width+offset+i*features+d];break;}}
  }
}
__aicore__ inline void guard(__gm__ I* owners,__gm__ float* weights,__gm__ float* bias,
    __gm__ float* values,__gm__ uint8_t* roots,__gm__ float* clean,__gm__ uint8_t* flags,
    __gm__ int32_t* error,I rows,I size,I features,I mode){
  if(error[0])return;
  for(I r=0;r<rows;++r){const bool active=owners[r]>=0;
    if(mode==0){
      for(I i=0;i<size*features;++i){auto p=weights+r*size*features+i;if(!active)*p=0;else if(!finite(*p)){error[0]=27;return;}}
      for(I i=0;i<features;++i){auto p=bias+r*features+i;if(!active)*p=0;else if(!finite(*p)){error[0]=27;return;}}
    }else if(mode==1){for(I i=0;i<features;++i){auto p=values+r*features+i;if(!active)*p=0;else if(!finite(*p)){error[0]=27;return;}}}
    else {flags[r]=active&&roots[r];for(I i=0;i<features;++i){clean[r*features+i]=0;
      if(flags[r]){if(!finite(values[r*features+i])){error[0]=28;return;}clean[r*features+i]=values[r*features+i];}}}
  }
}
__aicore__ inline void reverse(__gm__ I* owners,__gm__ I* meta,__gm__ I* mapping,
    __gm__ float* taps_bar,__gm__ float* weights_bar,__gm__ float* bias_bar,__gm__ uint8_t* roots,
    __gm__ float* old_bar,__gm__ float* current_bar,__gm__ uint8_t* cache_roots,
    __gm__ float* weight_out,__gm__ float* bias_out,__gm__ uint8_t* owner_roots,__gm__ int32_t* error,
    I rows,I raw_width,I capacity,I offset,I features,I size,I banks){
  if(error[0])return;
  for(I r=0;r<rows;++r){cache_roots[r]=roots[r]&&size>1&&meta[r*(4+capacity)+2]>0;
    for(I i=0;i<raw_width;++i)old_bar[r*raw_width+i]=0;
    for(I d=0;d<features;++d)current_bar[r*features+d]=roots[r]?taps_bar[r*size*features+d]:0;
    if(!roots[r])continue;
    for(I k=0;k<size;++k)for(I d=0;d<features;++d){const float x=taps_bar[(r*size+k)*features+d];
      // An absent historical tap is not a raw-cache argument. Its temporary
      // product can overflow even while every actual returned VJP is finite.
      if((k==0||mapping[r*size+k]>=0)&&!finite(x)){error[0]=28;return;}
      if(k&&mapping[r*size+k]>=0)old_bar[r*raw_width+offset+mapping[r*size+k]*features+d]=x;}
  }
  for(I bank=0;bank<banks;++bank){owner_roots[bank]=0;
    for(I i=0;i<size*features;++i)weight_out[bank*size*features+i]=0;
    for(I i=0;i<features;++i)bias_out[bank*features+i]=0;
    for(I r=0;r<rows;++r)if(roots[r]&&owners[r]==bank){owner_roots[bank]=1;
      for(I i=0;i<size*features;++i)weight_out[bank*size*features+i]+=weights_bar[r*size*features+i];
      for(I i=0;i<features;++i)bias_out[bank*features+i]+=bias_bar[r*features+i];}
    for(I i=0;i<size*features;++i)if(!finite(weight_out[bank*size*features+i])){error[0]=28;return;}
    for(I i=0;i<features;++i)if(!finite(bias_out[bank*features+i])){error[0]=28;return;}
  }
}
}
