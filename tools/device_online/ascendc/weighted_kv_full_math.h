#pragma once
#include "weighted_numeric_time.h"
namespace tide_weighted_full_device {
using namespace tide_weighted_device;
__aicore__ inline void plan(__gm__ uint8_t* active,__gm__ I* owners,__gm__ I* output,__gm__ int32_t* error,I rows,I banks){
  if(error[0])return;for(I r=0;r<rows;++r){output[r]=-1;if(!active[r])continue;
    if(owners[r]<0||owners[r]>=banks){error[0]=2;return;}output[r]=owners[r];}
}
__aicore__ inline void bias(__gm__ I* ages,__gm__ float* output,__gm__ int32_t* error,I count,float scale,float slope){
  if(error[0])return;for(I i=0;i<count;++i){const float value=-tide_weighted_time_device::as_float(ages[i])/scale*slope;
    if(!finite(value)){error[0]=33;return;}output[i]=value;}
}
__aicore__ inline void roots(__gm__ uint8_t* root,__gm__ uint8_t* flags,__gm__ uint8_t* pool,
    __gm__ uint8_t* query,__gm__ uint8_t* window,__gm__ int32_t* error,I rows,I heads,I mode){
  if(error[0])return;for(I r=0;r<rows;++r){if(mode==0){pool[2*r]=root[r];pool[2*r+1]=0;continue;}
    bool weight=false,key=false,value=false;for(I h=0;h<heads;++h){const I i=(h*rows+r)*3;
      weight=weight||flags[i];value=value||flags[i+1];key=key||flags[i+2];}
    query[r]=key;window[5*r]=key;window[5*r+1]=value;window[5*r+2]=0;window[5*r+3]=0;window[5*r+4]=weight;
  }
}
}
