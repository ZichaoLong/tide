#pragma once
#include "weighted_allocation_math.h"
#include "../weighted_region_layout.h"
namespace tide_weighted_region_device {
using namespace tide_weighted_device;
using namespace tide::device_online::weighted_region;
// Complete candidate reductions. Neural heads are supplied by batched vendor
// operators between this stage and finish; no host reads of these features.
__aicore__ inline int prepare(Math& math,__gm__ float* w,__gm__ float* mu,__gm__ float* s,
    __gm__ float* evidence,__gm__ float* current,I n,I ewidth,I qwidth,bool context,bool attention,
    bool query_received,__gm__ float* ctx,__gm__ float* qry,__gm__ float* stats){
  float received=0,mass=0,query_mass=0;
  for(I j=0;j<n;++j){
    if(!finite(w[j])||!finite(mu[j])||!finite(s[j])||w[j]<0||w[j]>mu[j]||mu[j]>1)return 20;
    received+=w[j];mass+=mu[j];query_mass+=query_received?w[j]:mu[j];
    if(context)for(I d=0;d<ewidth;++d){const float x=evidence[j*ewidth+d];
      if(!finite(x))return 20;ctx[d]+=w[j]*x;}
    if(attention)for(I d=0;d<qwidth;++d){const float x=current[j*qwidth+d];
      if(!finite(x))return 20;qry[d]+=(query_received?w[j]:mu[j])*x;}
  }
  stats[ReceivedMass]=received;stats[CapMass]=mass;stats[QueryMass]=query_mass;
  if(context){
    for(I d=0;d<ewidth;++d)ctx[d]/=maximum(1,received);
    ctx[ewidth]=math.log1p(received);ctx[ewidth+1]=math.log1p(mass);
    for(I d=0;d<ewidth+2;++d)if(!finite(ctx[d]))return 20;
  }
  if(attention)for(I d=0;d<qwidth;++d){qry[d]/=maximum(1,query_mass);if(!finite(qry[d]))return 20;}
  return 0;
}
__aicore__ inline int finish(Math& math,__gm__ float* local_score,__gm__ float* key,I n,I qwidth,
    I mode,bool context,bool attention,I k,float fixed,float tau,float alpha_local,float alpha_query,
    float inverse_sqrt,__gm__ float* stats,__gm__ float* head,__gm__ float* query,
    __gm__ float* scores,__gm__ float* bounds,__gm__ float* parameters){
  const bool total=mode==Total||mode==ZeroTotal,threshold=mode==Threshold||mode==HardThreshold
    ||mode==Constrained||mode==Hard;
  const float shift=threshold?(context?head[0]:fixed):0;
  if((total||threshold)&&context&&!finite(head[0]))return 20;
  if(total){
    const float rho=context?math.sigmoid(head[0]):fixed;
    const float lower=minimum(1,stats[ReceivedMass]),upper=minimum(float(k),stats[CapMass]);
    const float bound=lower+rho*(upper-lower);bounds[0]=bound;bounds[1]=bound;parameters[0]=bound;
  }else if(threshold){bounds[0]=minimum(1,stats[mode<=HardThreshold?ReceivedMass:CapMass]);bounds[1]=float(k);}
  bounds[2]=tau;bounds[3]=shift;parameters[1]=tau;
  for(I j=0;j<n;++j){float value=local_score[j];
    if(attention){float dot=0;
      for(I d=0;d<qwidth;++d){if(!finite(key[j*qwidth+d])||!finite(query[d]))return 20;dot+=key[j*qwidth+d]*query[d];}
      value=alpha_local*value+alpha_query*dot*inverse_sqrt;
    }
    if(mode==HardThreshold||mode==Hard)value-=shift;
    if(!finite(value))return 20;scores[j]=value;
  }
  return 0;
}
}
