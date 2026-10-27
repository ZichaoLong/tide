#include "weighted_allocation_math.h"
#include "../weighted_sending_layout.h"
namespace {
using namespace tide_weighted_device;
using namespace tide::device_online::weighted_sending;
using tide::device_online::weighted_allocation::GradientOverflow;
__aicore__ inline int backward(Math& math,__gm__ float* d,__gm__ float* stats,__gm__ float* g,
    I n,I width,I branch,I kind,I base,I k,__gm__ float* v,__gm__ float* scalars,__gm__ uint8_t* flags){
  for(I j=0;j<n;++j)if(!finite(g[j]))return GradientOverflow;
  if(kind>=SortedFill){
    flags[Caps]=1;flags[Bound]=kind==SortedFill;
    const float partial=branch>=0?g[branch]:0;
    if(kind==SortedFill)scalars[Total]=partial;
    for(I j=0;j<n;++j)v[width+j]=g[j]*d[j*Details+Share]-partial*d[j*Details+LogRatio];
  }else if(kind==Proportional){
    flags[Caps]=1;float mean=0;
    if(stats[Mass]>float(k)){for(I j=0;j<n;++j)mean+=g[j]*d[j*Details+Cap];mean/=stats[Mass];}
    for(I j=0;j<n;++j)v[width+j]=stats[Scale]*(g[j]-mean);
  }else{
    flags[Scores]=1;flags[Temp]=1;
    flags[Caps]=kind==Direct||kind==Pooled||base==CapBase;flags[Received]=base==ReceivedBase;
    if(branch==0)return 0; // Declared connected-zero map at all-zero base.
    float mean=0,bound_grad=0,temp=0;
    for(I j=0;j<n;++j){const float p=d[j*Details+Share],factor=kind==Direct?d[j*Details+Cap]:stats[Scale];
      mean+=p*g[j]*factor;bound_grad+=p*g[j];}
    for(I j=0;j<n;++j){
      const float p=d[j*Details+Share],factor=kind==Direct?d[j*Details+Cap]:stats[Scale];
      const float centered=g[j]*factor-mean;
      const float ds=p*centered/stats[Tau];
      const float da=centered==0?0:(centered>0?1:-1)*math.exp(d[j*Details+LogRatio]+math.log(magnitude(centered)));
      v[j]=ds;
      if(base==CapBase)v[width+j]=da;
      else if(base==ReceivedBase)v[2*width+j]=da;
      if(kind==Direct)v[width+j]+=g[j]*p;
      if(kind==Pooled&&stats[Mass]<=1)v[width+j]+=bound_grad;
      if(p>0)temp-=ds*d[j*Details+Centered];
    }
    scalars[Temperature]=temp;
  }
  for(I j=0;j<n;++j)for(I q=0;q<3;++q)if(!finite(v[q*width+j]))return GradientOverflow;
  for(I q=0;q<Parameters;++q)if(!finite(scalars[q]))return GradientOverflow;
  return 0;
}
}
extern "C" __global__ __aicore__ void tide_weighted_sending_vjp(GM_ADDR lengths,GM_ADDR branches,
    GM_ADDR details,GM_ADDR statistics,GM_ADDR cotangent,GM_ADDR roots,GM_ADDR vectors,GM_ADDR scalars,
    GM_ADDR connections,GM_ADDR row_errors,GM_ADDR error,int64_t rows,int64_t width,int64_t kind,int64_t base,int64_t k){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)lengths);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;Math math;math.init();
  for(I row=AscendC::GetBlockIdx();row<rows;row+=AscendC::GetBlockNum()){
    auto v=(__gm__ float*)vectors+row*3*width,s=(__gm__ float*)scalars+row*Parameters;
    auto flags=(__gm__ uint8_t*)connections+row*Connections;auto e=(__gm__ int32_t*)row_errors+row;
    for(I j=0;j<3*width;++j)v[j]=0;for(I j=0;j<Parameters;++j)s[j]=0;
    for(I j=0;j<Connections;++j)flags[j]=0;*e=0;
    const I n=((__gm__ I*)lengths)[row];if(n<0||n>width){*e=2;continue;}
    if(!n||!((__gm__ uint8_t*)roots)[row])continue;
    *e=backward(math,(__gm__ float*)details+row*width*Details,(__gm__ float*)statistics+row*Statistics,
      (__gm__ float*)cotangent+row*width,n,width,((__gm__ I*)branches)[row],kind,base,k,v,s,flags);
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
