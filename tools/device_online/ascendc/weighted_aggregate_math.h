#pragma once
// 0770593 architecture §1/2: every physical atom participates independently.
#include "weighted_allocation_math.h"
namespace tide_weighted_aggregate_device {
using namespace tide_weighted_device;
__aicore__ inline int prepare(__gm__ float* payload,__gm__ I* owners,I n,I width,I d,float alpha,
    __gm__ float* clean,__gm__ I* active_owners,__gm__ float* total){
  *total=0;
  for(I j=0;j<width;++j){active_owners[j]=-1;for(I k=0;k<d;++k)clean[j*d+k]=0;}
  for(I j=0;j<n;++j){
    for(I k=0;k<d+1;++k)if(!finite(payload[j*(d+1)+k]))return 20;
    *total+=maximum(0,payload[j*(d+1)]);
  }
  if(!finite(*total))return 20;
  if(*total>0)for(I j=0;j<n;++j){
    for(I k=0;k<d;++k)clean[j*d+k]=payload[j*(d+1)+k+1];
    if(alpha>0){if(owners[j]<0)return 2;active_owners[j]=owners[j];}
  }
  return 0;
}
__aicore__ inline int reduce(__gm__ float* payload,__gm__ float* transformed,I n,I width,I d,float alpha,
    float total,__gm__ float* values,__gm__ float* output){
  output[0]=total;for(I k=0;k<d;++k)output[k+1]=0;
  for(I j=0;j<width*d;++j)values[j]=0;
  if(total==0)return 0;
  for(I j=0;j<n;++j){const float mass=maximum(0,payload[j*(d+1)]);
    for(I k=0;k<d;++k){const float z=payload[j*(d+1)+k+1];
      const float value=alpha>0?z+alpha*transformed[j*d+k]:z;
      if(!finite(value))return 20;values[j*d+k]=value;output[k+1]+=mass*value;
    }
  }
  for(I k=0;k<d;++k){output[k+1]/=total;if(!finite(output[k+1]))return 20;}
  return 0;
}
// The public payload root is the complete (mass, content) tensor. Its presence
// follows the fiber cotangent, including the zero-total mass-only branch.
__aicore__ inline int reverse_mean(__gm__ float* payload,__gm__ float* values,__gm__ float* output,
    __gm__ float* bar,I n,I width,I d,float alpha,bool root,__gm__ float* gradient,
    __gm__ float* transform_gradient,__gm__ uint8_t* connected,__gm__ uint8_t* source_roots){
  for(I j=0;j<width;++j){connected[j]=0;source_roots[j]=0;
    for(I k=0;k<d+1;++k)gradient[j*(d+1)+k]=0;
    for(I k=0;k<d;++k)transform_gradient[j*d+k]=0;
  }
  if(!root||!n)return 0;
  for(I k=0;k<d+1;++k)if(!finite(bar[k]))return 23;
  const float total=output[0];
  for(I j=0;j<n;++j){const float raw=payload[j*(d+1)],mass=maximum(0,raw);float gm=bar[0];
    connected[j]=1;source_roots[j]=total>0&&alpha>0;
    if(total>0)for(I k=0;k<d;++k){
      // Exact singleton mean has a connected-zero mass derivative.
      if(n>1)gm+=bar[k+1]*(values[j*d+k]-output[k+1])/total;
      const float gz=(mass/total)*bar[k+1];gradient[j*(d+1)+k+1]=gz;
      transform_gradient[j*d+k]=alpha*gz;
      if(!finite(gz)||!finite(transform_gradient[j*d+k]))return 23;
    }
    gradient[j*(d+1)]=raw>=0?gm:0;
    if(!finite(gradient[j*(d+1)]))return 23;
  }
  return 0;
}
// Architecture §1 declared normalized-coordinate RMS adjoint. Actual vendor
// normalization and reciprocal standard deviation are saved by the tape.
__aicore__ inline int reverse_norm(__gm__ float* normalized,__gm__ float* rstd,__gm__ float* bar,
    __gm__ uint8_t* roots,I n,I d,__gm__ float* gradient){
  for(I j=0;j<n;++j)if(roots[j]){float contraction=0;
    for(I k=0;k<d;++k)contraction+=bar[j*d+k]*normalized[j*d+k];
    contraction/=float(d);
    for(I k=0;k<d;++k){gradient[j*(d+1)+k+1]+=(bar[j*d+k]-normalized[j*d+k]*contraction)*rstd[j];
      if(!finite(gradient[j*(d+1)+k+1]))return 23;}
  }
  return 0;
}
}
