#pragma once
#include "weighted_numeric_time.h"
namespace tide_weighted_recurrent_device {
using namespace tide_weighted_device;
// Fixed fields: value, H, Z, W, optional conv_values. All padding is sanitized.
__aicore__ inline void prepare(__gm__ float* raw,__gm__ I* meta,__gm__ I* time,__gm__ float* mass,
    __gm__ I* owners,__gm__ uint8_t* present,__gm__ float* rates,__gm__ float* clean,__gm__ float* decay,
    __gm__ I* selected,__gm__ int32_t* error,I rows,I width,I capacity,I d,I matrices,I matrix_width,I ds,I cf,I banks,I positive,I horizon){
  if(error[0])return;Math math;math.init();const I iw=4+capacity,hoff=d,zoff=hoff+matrices*matrix_width,woff=zoff+2*ds,coff=woff+2;
  for(I r=0;r<rows;++r){selected[r]=-1;for(I j=0;j<width;++j){clean[r*width+j]=0;decay[r*width+j]=0;}
    if(!present[r])continue;if(owners[r]<0||owners[r]>=banks||time[r]<0||(positive&&!finite(mass[r]))){error[0]=38;return;}
    if(positive&&mass[r]<=0)continue;
    const I m=r*iw,last=meta[m],observed=meta[m+1],n=meta[m+2],stamp=meta[m+3];
    if(last< -1||observed<0||n<0||n>capacity||n>observed||stamp!=(last<0?0:last)||((observed==0)!=(last==-1))||(!cf&&n)||(cf&&n>horizon+1)){
      error[0]=38;return;}
    for(I i=0;i<n;++i)if(meta[m+4+i]<0||meta[m+4+i]>last||(i&&meta[m+4+i]<=meta[m+3+i])){error[0]=38;return;}
    const I age=time[r]>stamp?time[r]-stamp:0;const float delta=tide_weighted_time_device::as_float(age);selected[r]=owners[r];
    for(I j=0;j<coff+n*cf;++j){const float x=raw[r*width+j];if(!finite(x)||(j<d&&x!=0)||(j>=woff&&j<coff&&x<0)){error[0]=38;return;}
      clean[r*width+j]=x;decay[r*width+j]=1;
    }
    for(I h=0;h<matrices;++h){const float factor=math.exp(-rates[h]*delta);
      for(I j=0;j<matrix_width;++j)decay[r*width+hoff+h*matrix_width+j]=factor;}
    for(I h=0;h<2;++h){const float factor=math.exp(-rates[matrices+h]*delta);
      for(I j=0;j<ds;++j)decay[r*width+zoff+h*ds+j]=factor;decay[r*width+woff+h]=factor;}
  }
}
// Combines separately rooted effective and identity paths. No raw reread.
__aicore__ inline void reverse(__gm__ I* selected,__gm__ I* meta,__gm__ float* decay,__gm__ I* fields,
    __gm__ float* bars,__gm__ uint8_t* roots,__gm__ float* bypass,__gm__ uint8_t* bypass_roots,
    __gm__ float* output,__gm__ uint8_t* connected,__gm__ int32_t* error,I rows,I width,I capacity,I count){
  if(error[0])return;
  for(I r=0;r<rows;++r){for(I j=0;j<width;++j)output[r*width+j]=0;
    for(I f=0;f<count;++f){const I index=r*count+f,off=fields[3*f];
      const bool a=selected[r]>=0&&roots[index],b=bypass_roots[index];connected[index]=a||b;
      if(!a&&!b)continue;const I n=fields[3*f+1]*(fields[3*f+2]?meta[r*(4+capacity)+2]:1);
      for(I j=0;j<n;++j){const I at=r*width+off+j;float v=0;
        if(a)v=bars[at]*decay[at];if(b)v+=bypass[at];if(!finite(v)){error[0]=39;return;}output[at]=v;}
    }
  }
}
}
