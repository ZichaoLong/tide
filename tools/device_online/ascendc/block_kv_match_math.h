#pragma once
#include "weighted_numeric_time.h"
namespace tide_block_kv_match_device {
using namespace tide_weighted_device;
// Relative phases come from exact int64 differences. No absolute-time rotation.
__aicore__ inline void phase(__gm__ I* ages,__gm__ uint8_t* visible,__gm__ float* frequencies,
    __gm__ float* output,__gm__ float* bias,__gm__ int32_t* error,I count,I pairs,float scale,float slope){
  if(error[0])return;for(I i=0;i<count;++i){
    const float delta=visible[i]?-tide_weighted_time_device::as_float(ages[i])/scale:0;
    const float b=delta*slope;if(!finite(b)){error[0]=84;return;}bias[i]=b;
    for(I j=0;j<pairs;++j){const float x=delta*frequencies[j];if(!finite(x)){error[0]=84;return;}output[i*pairs+j]=x;}}
}
__aicore__ inline void match(__gm__ float* query,__gm__ float* keys,__gm__ uint8_t* visible,
    __gm__ float* cosine,__gm__ float* sine,__gm__ float* logits,__gm__ float* bars,
    __gm__ float* dq,__gm__ float* dk,__gm__ int32_t* error,I owners,I queries,I entries,I width,I pairs,I reverse,float scale){
  if(error[0])return;
  if(reverse){for(I i=0;i<owners*queries*width;++i)dq[i]=0;for(I i=0;i<owners*entries*width;++i)dk[i]=0;}
  for(I o=0;o<owners;++o)for(I q=0;q<queries;++q)for(I k=0;k<entries;++k){const I a=(o*queries+q)*entries+k,qi=(o*queries+q)*width,ki=(o*entries+k)*width;
    if(!visible[a]){if(!reverse)logits[a]=0;continue;}const float g=reverse?bars[a]*scale:0;float dot=0;
    if(reverse&&!finite(g)){error[0]=85;return;}
    for(I j=0;j<width-2*pairs;++j){if(reverse){dq[qi+j]+=g*keys[ki+j];dk[ki+j]+=g*query[qi+j];}else dot+=query[qi+j]*keys[ki+j];}
    for(I j=0;j<pairs;++j){const I pos=width-2*pairs+j*2;const float c=cosine[a*pairs+j],s=sine[a*pairs+j];
      const float x=keys[ki+pos],y=keys[ki+pos+1],u=x*c-y*s,v=x*s+y*c;
      if(reverse){dq[qi+pos]+=g*u;dq[qi+pos+1]+=g*v;dk[ki+pos]+=g*(query[qi+pos]*c+query[qi+pos+1]*s);
        dk[ki+pos+1]+=g*(-query[qi+pos]*s+query[qi+pos+1]*c);}
      else dot+=query[qi+pos]*u+query[qi+pos+1]*v;
    }
    if(!reverse){if(!finite(dot)){error[0]=84;return;}logits[a]=dot*scale;}
  }
}
}
