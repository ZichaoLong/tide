#pragma once
#include "weighted_numeric_time.h"
namespace tide_block_kv_read_device {
using namespace tide_weighted_device;
__aicore__ inline void plan(__gm__ I* mapping,__gm__ I* meta,__gm__ float* bank_weights,__gm__ I* stamps,
    __gm__ I* times,__gm__ float* input,__gm__ I* owners,__gm__ uint8_t* present,
    __gm__ float* weights,__gm__ float* decay,__gm__ uint8_t* visible,__gm__ I* ages,__gm__ uint8_t* selected,
    __gm__ I* qowners,__gm__ I* eowners,__gm__ I* fowners,__gm__ float* clean,__gm__ int32_t* error,
    I batch,I length,I nodes,I capacity,I entries,I width,I banks,I horizon,I query,I full,float beta){
  if(error[0])return;Math math;math.init();const I rows=batch*length*nodes,extent=rows*entries;
  for(I e=0;e<batch*nodes*entries;++e)eowners[e]=-1;
  for(I b=0;b<batch;++b)for(I n=0;n<nodes;++n)for(I q=0;q<length;++q){
    const I owner=b*nodes+n,r=(b*length+q)*nodes+n,a=owner*length+q;
    qowners[r]=-1;fowners[r]=-1;selected[a]=present[r];for(I d=0;d<width;++d)clean[r*width+d]=0;
    for(I k=0;k<entries;++k){const I i=a*entries+k;visible[i]=0;ages[i]=0;weights[i]=weights[extent+i]=0;decay[i]=decay[extent+i]=0;}
    if(!present[r])continue;
    const I size=meta[r*(capacity+4)+2],time=times[r];if(time<0||size<0||size>capacity){error[0]=83;return;}
    if(full){if(owners[r]<0||owners[r]>=banks){error[0]=83;return;}fowners[r]=owners[r];}
    bool positive=false;
    for(I j=0;j<size;++j){const I e=mapping[r*capacity+j];if(e<owner*entries||e>=(owner+1)*entries){error[0]=83;return;}
      const I stamp=stamps[e];if(stamp>time)continue;const I age=time-stamp;if(horizon&&age>=horizon)continue;
      const I i=a*entries+e-owner*entries;const float w=bank_weights[e],short_decay=math.exp(-tide_weighted_time_device::as_float(age)*beta);
      if(!finite(w)||w<0||w>1||!finite(short_decay)||short_decay<0||short_decay>1){error[0]=83;return;}
      ages[i]=age;visible[i]=1;decay[i]=short_decay;decay[extent+i]=1;weights[i]=w*short_decay;weights[extent+i]=w;positive=positive||w>0;
    }
    if(!query||!positive)continue;
    if(owners[r]<0||owners[r]>=banks){error[0]=83;return;}qowners[r]=owners[r];
    for(I d=0;d<width;++d){const float x=input[r*width+d];if(!finite(x)){error[0]=83;return;}clean[r*width+d]=x;}
    for(I k=0;k<entries;++k)if(weights[extent+a*entries+k]>0){const I e=owner*entries+k;
      // Sharing is legal only for the same parameter owner and actual entry.
      if(eowners[e]>=0&&eowners[e]!=owners[r]){error[0]=83;return;}eowners[e]=owners[r];}
  }
}
}
