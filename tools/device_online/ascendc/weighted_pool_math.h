#pragma once
#include "weighted_allocation_math.h"
namespace tide_weighted_pool_device {
using namespace tide_weighted_device;
// meta[R,3]: presence, length, positive count. Numerical zeros never encode absence.
__aicore__ inline void prepare(__gm__ float* w,__gm__ float* v,__gm__ float* logits,
    __gm__ I* lengths,__gm__ uint8_t* present,__gm__ float* clean_w,__gm__ float* clean_v,
    __gm__ float* log_scores,__gm__ I* meta,__gm__ int32_t* error,I rows,I capacity,I width,I attention){
  if(error[0])return;Math math;math.init();
  for(I r=0;r<rows;++r){const I n=present[r]?lengths[r]:0;
    if(n<0||n>capacity){error[0]=29;return;}
    meta[3*r]=present[r];meta[3*r+1]=n;I positives=0;
    for(I i=0;i<capacity;++i){const I a=r*capacity+i;const float weight=i<n?w[a]:0;
      if(!finite(weight)||weight<0){error[0]=29;return;}
      clean_w[a]=weight;if(weight>0)++positives;
      for(I d=0;d<width;++d){const float value=i<n&&(!attention||weight>0)?v[a*width+d]:0;
        if(!finite(value)){error[0]=29;return;}clean_v[a*width+d]=value;}
      log_scores[a]=-inf();if(attention&&weight>0){const float score=logits[a]+math.log(weight);
        if(!finite(logits[a])||!finite(score)){error[0]=29;return;}log_scores[a]=score;}
    }
    meta[3*r+2]=positives;
    if(!positives)for(I i=0;i<capacity;++i)log_scores[r*capacity+i]=0;
  }
}
__aicore__ inline void scale(__gm__ float* w,__gm__ float* total,__gm__ float* prob,
    __gm__ I* meta,__gm__ float* coeff,__gm__ float* amplitude,__gm__ int32_t* error,
    I rows,I capacity,I attention){
  if(error[0])return;
  for(I r=0;r<rows;++r){const float sum=total[r];if(!finite(sum)||sum<0){error[0]=29;return;}
    amplitude[r]=attention?(meta[3*r+2]?minimum(1,sum):0):1;
    for(I i=0;i<capacity;++i){const I a=r*capacity+i;
      const float value=attention?(meta[3*r+2]&&w[a]>0?prob[a]:0):w[a]/maximum(1,sum);
      if(!finite(value)||value<0){error[0]=29;return;}coeff[a]=value;
      if(attention)prob[a]=value; // Empty rows must have exactly zero mean.
    }
  }
}
__aicore__ inline void reverse_prepare(__gm__ I* meta,__gm__ float* weights,__gm__ float* values,
    __gm__ float* prob,__gm__ float* coeff,__gm__ float* total,__gm__ float* output,
    __gm__ float* vector_bar,__gm__ float* total_bar,__gm__ uint8_t* roots,
    __gm__ float* clean_bar,__gm__ float* clean_total,__gm__ float* centered,__gm__ float* value_coeff,
    __gm__ uint8_t* connected,__gm__ int32_t* error,I rows,I capacity,I width,I attention){
  if(error[0])return;
  for(I r=0;r<rows;++r){const bool vector=meta[3*r]&&roots[2*r]&&(!attention||meta[3*r+2]),stat=meta[3*r]&&roots[2*r+1];
    connected[3*r]=vector||stat;connected[3*r+1]=vector;connected[3*r+2]=vector&&attention;
    clean_total[r]=stat?total_bar[r]:0;if(!finite(clean_total[r])){error[0]=30;return;}
    I first=0;if(attention&&meta[3*r+2])while(weights[r*capacity+first]<=0)++first;
    for(I d=0;d<width;++d){const float g=vector?vector_bar[r*width+d]:0;
      if(!finite(g)){error[0]=30;return;}clean_bar[r*width+d]=g;
      // Translation-invariant Attention derivative, as declared by attention_mean.
      float mean=0,anchor=0;if(attention&&vector){anchor=values[(r*capacity+first)*width+d];
        for(I i=0;i<meta[3*r+1];++i)if(weights[r*capacity+i]>0)
          mean+=prob[r*capacity+i]*(values[(r*capacity+i)*width+d]-anchor);}
      for(I i=0;i<capacity;++i){const I a=r*capacity+i;float x=0;
        if(vector&&i<meta[3*r+1]&&(!attention||weights[a]>0))
          x=attention?(values[a*width+d]-anchor-mean):(values[a*width+d]-(total[r]>=1?output[r*width+d]:0));
        centered[a*width+d]=x;
      }
    }
    for(I i=0;i<capacity;++i)value_coeff[r*capacity+i]=vector?coeff[r*capacity+i]*(attention?minimum(1,total[r]):1):0;
  }
}
__aicore__ inline void reverse_finish(__gm__ I* meta,__gm__ float* w,__gm__ float* prob,
    __gm__ float* total,__gm__ float* dots,__gm__ float* mean_dot,__gm__ float* stat,
    __gm__ uint8_t* connected,__gm__ float* dw,__gm__ float* dv,__gm__ float* dl,
    __gm__ int32_t* error,I rows,I capacity,I width,I attention){
  if(error[0])return;
  for(I r=0;r<rows;++r){const bool vector=connected[3*r+1];const float sum=total[r];
    for(I i=0;i<capacity;++i){const I a=r*capacity+i;float weight=0,score=0;
      if(i<meta[3*r+1]){weight=stat[r];if(vector){
        if(attention){if(sum<=1)weight+=mean_dot[r];if(w[a]>0){score=minimum(1,sum)*prob[a]*dots[a];weight+=score/w[a];}}
        else weight+=dots[a]/maximum(1,sum);
      }}
      dw[a]=weight;dl[a]=score;
      if(!finite(weight)||!finite(score)){error[0]=30;return;}
      for(I d=0;d<width;++d)if(!finite(dv[a*width+d])){error[0]=30;return;}
    }
  }
}
}
