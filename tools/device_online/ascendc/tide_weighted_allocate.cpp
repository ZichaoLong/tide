#include "weighted_allocation_math.h"
namespace {
using namespace tide_weighted_device;
using namespace tide::device_online::weighted_allocation;
__aicore__ inline int solve(Response& rule,__gm__ float* s,__gm__ float* mu,__gm__ float* b,
    I n,__gm__ float* c,__gm__ float* slopes,__gm__ float* stats,__gm__ I* branch){
  const float low=b[Lo],high=b[Hi],tau=b[Temperature],shift=b[Shift];
  for(I j=0;j<BoundFields;++j)if(!finite(b[j]))return Domain;
  if(low<0||high<low||tau<=0)return Domain;
  float mass=0,center=-inf(),total=0;
  for(I j=0;j<n;++j){
    if(!finite(s[j])||!finite(mu[j])||mu[j]<0||mu[j]>1||!finite(s[j]-shift))return Domain;
    mass+=mu[j];center=maximum(center,s[j]-shift);
    total+=rule.value(s[j]-shift,mu[j],tau);
  }
  if(!finite(mass)||!finite(total)||low>mass)return Domain;
  if(mass==0||high==0){*branch=Zero;return 0;}
  if(low==mass){*branch=Caps;for(I j=0;j<n;++j)c[j]=mu[j];return 0;}
  *branch=low==high?Equality:total<low?Lower:total>high?Upper:Slack;
  float lam=0;
  if(*branch!=Slack){
    const float target=(*branch==Lower||*branch==Equality)?low:high;
    float lo=inf(),hi=-inf();
    for(I j=0;j<n;++j)if(mu[j]>0){
      const float point=(s[j]-shift-center)-rule.inverse(target/mass,mu[j],tau);
      lo=minimum(lo,point);hi=maximum(hi,point);
    }
    if(!finite(lo)||!finite(hi))return Domain;
    // Forward iterations only. The separately recorded response slopes define
    // the exact solution-map adjoint, not a derivative through these decisions.
    for(I iteration=0;iteration<80;++iteration){
      const float mid=lo+(hi-lo)/2;float sum=0;
      for(I j=0;j<n;++j)sum+=rule.value(s[j]-shift-center-mid,mu[j],tau);
      if(sum>target)lo=mid;else hi=mid;
    }
    lam=lo+(hi-lo)/2;
    if(rule.kind==Quad){
      float free_sum=0,capped=0;I count=0;
      for(I j=0;j<n;++j){
        const float x=s[j]-shift-center,v=rule.value(x-lam,mu[j],tau);
        if(v>0&&v<mu[j]){free_sum+=x;++count;}else if(v==mu[j])capped+=mu[j];
      }
      if(count)lam=(free_sum+tau*(capped-target))/float(count);
    }else{
      float best=inf(),chosen=lo;
      for(I k=0;k<3;++k){const float point=k==0?lo:k==1?lam:hi;float sum=0;
        for(I j=0;j<n;++j)sum+=rule.value(s[j]-shift-center-point,mu[j],tau);
        const float error=magnitude(sum-target);if(error<best){best=error;chosen=point;}}
      lam=chosen;
    }
    stats[0]=lam+center;
  }
  float sum=0,residual=0;
  for(I j=0;j<n;++j){
    const float x=*branch==Slack?s[j]-shift:s[j]-shift-center-lam;
    c[j]=rule.value(x,mu[j],tau);sum+=c[j];
    if(rule.support(x,mu[j],tau)&&c[j]==0)return SupportUnderflow;
    // Global multiplier is a report only; use the actual centered response.
    rule.slopes(x,mu[j],tau,c[j],slopes+j*SlopeFields);
    if(rule.kind==Quad)residual=maximum(residual,magnitude(c[j]-rule.value(x,mu[j],tau)));
  }
  residual=maximum(residual,maximum(low-sum,sum-high));stats[1]=residual;
  if(!finite(residual)||residual>64.f*float(n)*1.1920928955078125e-7f)return Residual;
  return 0;
}
}
extern "C" __global__ __aicore__ void tide_weighted_allocate(GM_ADDR scores,GM_ADDR caps,
    GM_ADDR bounds,GM_ADDR lengths,GM_ADDR controls,GM_ADDR slopes,GM_ADDR statistics,GM_ADDR branches,
    GM_ADDR row_errors,GM_ADDR error,int64_t rows,int64_t width,int64_t kind,float alpha){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)lengths);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  Response rule;rule.init(kind,alpha);
  for(I row=AscendC::GetBlockIdx();row<rows;row+=AscendC::GetBlockNum()){
    auto c=(__gm__ float*)controls+row*width,sl=(__gm__ float*)slopes+row*width*SlopeFields;
    auto st=(__gm__ float*)statistics+row*2;auto branch=(__gm__ I*)branches+row;
    auto e=(__gm__ int32_t*)row_errors+row;const I n=((__gm__ I*)lengths)[row];
    *e=0;*branch=Absent;st[0]=0;st[1]=0;
    for(I j=0;j<width;++j){c[j]=0;for(I k=0;k<SlopeFields;++k)sl[j*SlopeFields+k]=0;}
    if(n<0||n>width){*e=2;continue;}if(n==0)continue;
    if(kind<Quad||kind>Entropy||(kind==Entropy&&(!finite(alpha)||alpha<1||alpha>2))){*e=Domain;continue;}
    *e=solve(rule,(__gm__ float*)scores+row*width,(__gm__ float*)caps+row*width,
      (__gm__ float*)bounds+row*BoundFields,n,c,sl,st,branch);
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
