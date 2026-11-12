#pragma once
#include "weighted_allocation_math.h"
namespace tide_block_kv_pool_device {
using namespace tide_weighted_device;
__aicore__ inline void prepare(__gm__ float* weights,__gm__ float* logits,__gm__ uint8_t* visible,
    __gm__ uint8_t* present,__gm__ float* clean,__gm__ float* scores,__gm__ I* meta,
    __gm__ int32_t* error,I rows,I entries,I attention){
  if(error[0])return;Math math;math.init();
  for(I r=0;r<rows;++r){I positive=0;meta[r*2]=present[r];
    for(I k=0;k<entries;++k){const I a=r*entries+k;const float w=present[r]&&visible[a]?weights[a]:0;
      if(!finite(w)||w<0){error[0]=81;return;}clean[a]=w;scores[a]=-inf();
      if(w>0){++positive;if(attention){const float x=logits[a]+math.log(w);if(!finite(logits[a])||!finite(x)){error[0]=81;return;}scores[a]=x;}}
    }
    meta[r*2+1]=positive;if(!positive)for(I k=0;k<entries;++k)scores[r*entries+k]=0;
  }
}
__aicore__ inline void scale(__gm__ float* weights,__gm__ float* total,__gm__ float* probability,
    __gm__ I* meta,__gm__ float* coefficients,__gm__ float* amplitude,__gm__ int32_t* error,I rows,I entries,I attention){
  if(error[0])return;
  for(I r=0;r<rows;++r){const float w=total[r];if(!finite(w)||w<0){error[0]=81;return;}
    amplitude[r]=attention?(meta[r*2+1]?minimum(1,w):0):1;
    for(I k=0;k<entries;++k){const I a=r*entries+k;
      const float c=attention?(meta[r*2+1]&&weights[a]>0?probability[a]:0):weights[a]/maximum(1,w);
      if(!finite(c)||c<0){error[0]=81;return;}coefficients[a]=c;if(attention)probability[a]=c;}
  }
}
// One anchor per physical owner. Translation-invariant score derivatives do
// not subtract two large query dots; this also avoids a [O,Q,K,D] workspace.
__aicore__ inline void center(__gm__ float* values,__gm__ float* weights,__gm__ uint8_t* visible,__gm__ uint8_t* present,
    __gm__ float* clean,__gm__ float* centered,__gm__ float* anchor,__gm__ int32_t* error,
    I owners,I queries,I entries,I width,I attention){
  if(error[0])return;
  for(I o=0;o<owners;++o){I first=-1;
    for(I k=0;k<entries;++k){bool used=false;for(I q=0;q<queries;++q){const I a=(o*queries+q)*entries+k;used=used||(present[o*queries+q]&&visible[a]&&(!attention||weights[a]>0));}
      if(used&&first<0)first=k;
    }
    for(I d=0;d<width;++d){const float base=first<0?0:values[(o*entries+first)*width+d];
      if(!finite(base)){error[0]=81;return;}anchor[o*width+d]=base;
    }
    for(I k=0;k<entries;++k){bool used=false;for(I q=0;q<queries;++q){const I a=(o*queries+q)*entries+k;used=used||(present[o*queries+q]&&visible[a]&&(!attention||weights[a]>0));}
      for(I d=0;d<width;++d){const I a=(o*entries+k)*width+d;const float x=used?values[a]:0;
        if(!finite(x)){error[0]=81;return;}clean[a]=x;centered[a]=used?x-anchor[o*width+d]:0;}
    }
  }
}
__aicore__ inline void roots(__gm__ I* meta,__gm__ float* total,__gm__ float* vector_bar,
    __gm__ float* total_bar,__gm__ uint8_t* roots,__gm__ float* clean_bar,__gm__ float* scaled_bar,
    __gm__ float* stat,__gm__ uint8_t* connected,__gm__ int32_t* error,I rows,I width,I attention){
  if(error[0])return;
  for(I r=0;r<rows;++r){const bool vector=meta[r*2]&&roots[r*2]&&(!attention||meta[r*2+1]);
    const bool scalar=meta[r*2]&&roots[r*2+1];connected[r*3]=vector||scalar;connected[r*3+1]=vector;connected[r*3+2]=vector&&attention;
    const float s=scalar?total_bar[r]:0;if(!finite(s)){error[0]=82;return;}stat[r]=s;
    for(I j=0;j<width;++j){const float g=vector?vector_bar[r*width+j]:0;if(!finite(g)){error[0]=82;return;}
      clean_bar[r*width+j]=g;scaled_bar[r*width+j]=g*(attention?minimum(1,total[r]):1);}
  }
}
__aicore__ inline void finish(__gm__ float* weights,__gm__ float* probability,__gm__ float* total,
    __gm__ float* dots,__gm__ float* mean_dot,__gm__ float* centered_dot,__gm__ float* anchor_dot,
    __gm__ float* stat,__gm__ uint8_t* visible,
    __gm__ uint8_t* connected,__gm__ float* dw,__gm__ float* dl,__gm__ int32_t* error,I rows,I entries,I attention){
  if(error[0])return;
  for(I r=0;r<rows;++r)for(I k=0;k<entries;++k){const I a=r*entries+k;float g=0,s=0;
    if(visible[a]){g=stat[r];if(connected[r*3+1]){
      if(attention){if(total[r]<=1)g+=mean_dot[r];if(weights[a]>0){s=minimum(1,total[r])*probability[a]*(dots[a]-centered_dot[r]);g+=s/weights[a];}}
      else g+=(total[r]>=1?dots[a]-centered_dot[r]:dots[a]+anchor_dot[r])/maximum(1,total[r]);
    }}
    if(!finite(g)||!finite(s)){error[0]=82;return;}dw[a]=g;dl[a]=s;
  }
}
}
