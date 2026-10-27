#include "weighted_allocation_math.h"
#include "../weighted_sending_layout.h"
namespace {
using namespace tide_weighted_device;
using namespace tide::device_online::weighted_sending;
using tide::device_online::weighted_allocation::Domain;
using tide::device_online::weighted_allocation::SupportUnderflow;
__aicore__ inline float base_value(I base,float mu,float w){return base==CapBase?mu:base==ReceivedBase?w:1;}
__aicore__ inline bool earlier(I a,I b,__gm__ float* s,__gm__ I* ids){
  return b<0||s[a]>s[b]||(s[a]==s[b]&&ids[a]<ids[b]);
}
__aicore__ inline int send(Math& math,__gm__ float* s,__gm__ float* mu,__gm__ float* w,
    __gm__ float* params,__gm__ I* ids,I n,I kind,I base,I k,__gm__ float* c,
    __gm__ float* detail,__gm__ float* stats,__gm__ I* branch){
  float mass=0;
  for(I j=0;j<n;++j){
    if(!finite(mu[j])||mu[j]<0||mu[j]>1)return Domain;
    if(kind!=Proportional&&!finite(s[j]))return Domain;
    if(kind<=Pooled&&(!finite(w[j])||w[j]<0||w[j]>mu[j]))return Domain;
    if(kind>=SortedFill){
      if(ids[j]<0)return Domain;
      for(I i=0;i<j;++i)if(ids[i]==ids[j])return Domain;
    }
    mass+=mu[j];detail[j*Details+Cap]=mu[j];
  }
  if(!finite(mass))return Domain;stats[Mass]=mass;
  if(kind==Proportional){
    const float scale=mass<=float(k)?1:float(k)/mass;stats[Scale]=scale;
    for(I j=0;j<n;++j){c[j]=mu[j]*scale;if(mu[j]>0&&c[j]==0)return SupportUnderflow;}
    return 0;
  }
  if(kind>=SortedFill){
    float remaining=params[Total];
    if(kind==SortedFill&&(!finite(remaining)||remaining<0||remaining>mass))return Domain;
    *branch=-1;
    if(kind==SortedFill&&remaining==mass){
      for(I j=0;j<n;++j){c[j]=mu[j];detail[j*Details+Share]=mu[j]>0?1.f:0.f;}return 0;
    }
    for(I count=0;count<n;++count){
      I best=-1;
      for(I j=0;j<n;++j)if(mu[j]>0&&detail[j*Details+Share]==0&&earlier(j,best,s,ids))best=j;
      if(best<0)break;
      if(kind==Hard){
        if(count>=k||(count>0&&s[best]<=0))break;
        c[best]=mu[best];detail[best*Details+Share]=1;
        if(s[best]<=0)break;
      }else{
        if(remaining<=0)break;
        if(remaining>=mu[best]){c[best]=mu[best];detail[best*Details+Share]=1;remaining-=mu[best];}
        else {c[best]=remaining;*branch=best;
          for(I j=0;j<n;++j)detail[j*Details+LogRatio]=detail[j*Details+Share];break;}
      }
    }
    return 0;
  }
  const float tau=params[Temperature];if(!finite(tau)||tau<=0)return Domain;stats[Tau]=tau;
  float center=-inf();bool positive=false;
  for(I j=0;j<n;++j)if(base_value(base,mu[j],w[j])>0){positive=true;center=maximum(center,s[j]);}
  *branch=positive;
  const float scale=kind==Pooled?minimum(1,mass):1;stats[Scale]=scale;
  if(!positive)return 0;
  float max_log=-inf();
  for(I j=0;j<n;++j){
    const float z=(s[j]-center)/tau,a=base_value(base,mu[j],w[j]);detail[j*Details+Centered]=z;
    if(a>0)max_log=maximum(max_log,z+math.log(a));
  }
  float total=0;
  for(I j=0;j<n;++j){const float a=base_value(base,mu[j],w[j]);
    if(a>0)total+=math.exp(detail[j*Details+Centered]+math.log(a)-max_log);}
  const float log_total=max_log+math.log(total);
  for(I j=0;j<n;++j){
    const float a=base_value(base,mu[j],w[j]),log_ratio=detail[j*Details+Centered]-log_total;
    const float p=a>0?math.exp(log_ratio+math.log(a)):0;
    detail[j*Details+Share]=p;detail[j*Details+LogRatio]=log_ratio;
    c[j]=p*(kind==Direct?mu[j]:scale);
    if(!finite(p)||!finite(c[j])||(a>0&&(p==0||c[j]==0)))return SupportUnderflow;
  }
  return 0;
}
}
extern "C" __global__ __aicore__ void tide_weighted_send(GM_ADDR scores,GM_ADDR caps,GM_ADDR received,
    GM_ADDR parameters,GM_ADDR ids,GM_ADDR lengths,GM_ADDR controls,GM_ADDR details,GM_ADDR statistics,
    GM_ADDR branches,GM_ADDR row_errors,GM_ADDR error,int64_t rows,int64_t width,int64_t kind,int64_t base,int64_t k){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)lengths);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  Math math;math.init();
  for(I row=AscendC::GetBlockIdx();row<rows;row+=AscendC::GetBlockNum()){
    auto c=(__gm__ float*)controls+row*width,d=(__gm__ float*)details+row*width*Details;
    auto st=(__gm__ float*)statistics+row*Statistics,params=(__gm__ float*)parameters+row*Parameters;
    auto branch=(__gm__ I*)branches+row;auto e=(__gm__ int32_t*)row_errors+row;
    for(I j=0;j<width;++j){c[j]=0;for(I q=0;q<Details;++q)d[j*Details+q]=0;}
    for(I q=0;q<Statistics;++q)st[q]=0;*branch=-1;*e=0;
    const I n=((__gm__ I*)lengths)[row];if(n<0||n>width){*e=2;continue;}if(!n)continue;
    if(kind<Direct||kind>Hard||((kind==Proportional||kind==Hard)?k<1:k!=0)
       ||(kind==Direct&&base!=CapBase&&base!=ReceivedBase)
       ||(kind==Unit&&base!=CapBase&&base!=UniformBase)||(kind>=Pooled&&base!=CapBase)){
      *e=Domain;continue;}
    *e=send(math,(__gm__ float*)scores+row*width,(__gm__ float*)caps+row*width,
      (__gm__ float*)received+row*width,params,(__gm__ I*)ids+row*width,n,kind,base,k,c,d,st,branch);
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
