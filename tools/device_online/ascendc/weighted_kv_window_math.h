#pragma once
#include "weighted_numeric_time.h"
namespace tide_weighted_window_device {
using namespace tide_weighted_device;
__aicore__ inline void prepare(__gm__ float* raw,__gm__ I* meta,__gm__ I* times,
    __gm__ uint8_t* present,__gm__ float* keys,__gm__ float* values,__gm__ float* summaries,
    __gm__ float* weights,__gm__ float* decay,__gm__ I* ages,__gm__ I* mapping,__gm__ I* lengths,
    __gm__ uint8_t* selected,__gm__ int32_t* error,I rows,I capacity,I d,I ds,I horizon,float beta){
  if(error[0])return;Math math;math.init();
  const I kw=d+1,vw=kw+capacity*d,pw=vw+capacity*d,ww=pw+capacity*ds,fw=ww+capacity,iw=capacity+4;
  for(I r=0;r<rows;++r){lengths[r]=0;selected[r]=present[r];
    for(I i=0;i<capacity;++i){const I a=r*capacity+i;ages[a]=0;mapping[a]=-1;
      for(I j=0;j<d;++j){keys[a*d+j]=0;values[a*d+j]=0;}for(I j=0;j<ds;++j)summaries[a*ds+j]=0;
      for(I channel=0;channel<2;++channel){weights[(r*2+channel)*capacity+i]=0;decay[(r*2+channel)*capacity+i]=0;}}
    if(!present[r])continue;
    const I n=meta[r*iw+2],last=meta[r*iw],count=meta[r*iw+1],time=times[r];
    if(time<0||last< -1||count<0||n<0||n>capacity||n>count||meta[r*iw+3]!=0||(horizon&&n>horizon)){error[0]=31;return;}
    for(I j=0;j<d+1;++j)if(raw[r*fw+j]!=0){error[0]=31;return;}
    I used=0;for(I i=0;i<n;++i){const I stamp=meta[r*iw+4+i];
      if(stamp<0||stamp>last||(i&&stamp<=meta[r*iw+3+i])){error[0]=31;return;}
      const float w=raw[r*fw+ww+i];if(!finite(w)||w<0||w>1){error[0]=31;return;}
      for(I j=0;j<d;++j)if(!finite(raw[r*fw+kw+i*d+j])||!finite(raw[r*fw+vw+i*d+j])){error[0]=31;return;}
      for(I j=0;j<ds;++j)if(!finite(raw[r*fw+pw+i*ds+j])){error[0]=31;return;}
      if(stamp>time)continue;const I age=time-stamp;if(horizon&&age>=horizon)continue;
      const I a=r*capacity+used;ages[a]=age;mapping[a]=i;
      for(I j=0;j<d;++j){keys[a*d+j]=raw[r*fw+kw+i*d+j];values[a*d+j]=raw[r*fw+vw+i*d+j];}
      for(I j=0;j<ds;++j)summaries[a*ds+j]=raw[r*fw+pw+i*ds+j];
      const float short_decay=math.exp(-tide_weighted_time_device::as_float(age)*beta);
      if(!finite(short_decay)||short_decay<0||short_decay>1){error[0]=31;return;}
      decay[r*2*capacity+used]=short_decay;decay[(r*2+1)*capacity+used]=1;
      weights[r*2*capacity+used]=w*short_decay;weights[(r*2+1)*capacity+used]=w;++used;
    }lengths[r]=used;
  }
}
__aicore__ inline void reverse(__gm__ I* mapping,__gm__ I* lengths,__gm__ uint8_t* present,
    __gm__ float* decay,__gm__ float* keys,__gm__ float* values,__gm__ float* summaries,
    __gm__ float* weights,__gm__ uint8_t* roots,__gm__ float* raw,__gm__ uint8_t* connected,
    __gm__ int32_t* error,I rows,I capacity,I d,I ds){
  if(error[0])return;const I kw=d+1,vw=kw+capacity*d,pw=vw+capacity*d,ww=pw+capacity*ds,fw=ww+capacity;
  for(I r=0;r<rows;++r){for(I j=0;j<fw;++j)raw[r*fw+j]=0;for(I j=0;j<5;++j)connected[5*r+j]=0;
    if(!present[r])continue;for(I j=0;j<3;++j)connected[5*r+j+1]=roots[5*r+j];connected[5*r+4]=roots[5*r+3]||roots[5*r+4];
    for(I i=0;i<lengths[r];++i){const I a=r*capacity+i,source=mapping[a];
      for(I role=0;role<3;++role)if(roots[5*r+role]){const I width=role==2?ds:d,offset=role==0?kw:(role==1?vw:pw);
        auto input=role==0?keys:(role==1?values:summaries);
        for(I j=0;j<width;++j){const float x=input[a*width+j];if(!finite(x)){error[0]=32;return;}raw[r*fw+offset+source*width+j]=x;}}
      float g=0;for(I channel=0;channel<2;++channel)if(roots[5*r+3+channel]){
        const I a=(2*r+channel)*capacity+i;const float x=weights[a];if(!finite(x)){error[0]=32;return;}g+=x*decay[a];}
      if(!finite(g)){error[0]=32;return;}raw[r*fw+ww+source]=g;
    }
  }
}
}
