#pragma once
#include "weighted_allocation_math.h"
namespace tide_weighted_query_device {
using namespace tide_weighted_device;
__aicore__ inline void plan(__gm__ float* weights,__gm__ I* lengths,__gm__ uint8_t* present,
    __gm__ I* owners,__gm__ float* input,__gm__ float* clean,__gm__ I* query_owners,
    __gm__ I* entry_owners,__gm__ int32_t* error,I rows,I capacity,I width,I banks){
  if(error[0])return;for(I r=0;r<rows;++r){query_owners[r]=-1;for(I d=0;d<width;++d)clean[r*width+d]=0;
    for(I i=0;i<capacity;++i)entry_owners[r*capacity+i]=-1;
    if(!present[r])continue;bool positive=false;
    for(I i=0;i<lengths[r];++i)positive=positive||weights[(r*2+1)*capacity+i]>0;
    if(!positive)continue;if(owners[r]<0||owners[r]>=banks){error[0]=2;return;}
    query_owners[r]=owners[r];for(I d=0;d<width;++d){const float x=input[r*width+d];if(!finite(x)){error[0]=20;return;}clean[r*width+d]=x;}
    for(I i=0;i<lengths[r];++i)if(weights[(r*2+1)*capacity+i]>0)entry_owners[r*capacity+i]=owners[r];
  }
}
__aicore__ inline void roots(__gm__ uint8_t* pool,__gm__ I* owners,__gm__ uint8_t* query,
    __gm__ uint8_t* keys,__gm__ uint8_t* values,__gm__ uint8_t* window,__gm__ int32_t* error,I rows,I capacity){
  if(error[0])return;for(I r=0;r<rows;++r){const bool k=pool[r*6+2]||pool[r*6+5],v=pool[r*6+1]||pool[r*6+4];
    query[r]=k;window[r*5]=k;window[r*5+1]=v;window[r*5+2]=0;window[r*5+3]=pool[r*6];window[r*5+4]=pool[r*6+3];
    for(I i=0;i<capacity;++i){keys[r*capacity+i]=k&&owners[r*capacity+i]>=0;values[r*capacity+i]=v&&owners[r*capacity+i]>=0;}
  }
}
}
