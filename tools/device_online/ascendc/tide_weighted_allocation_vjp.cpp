#include "weighted_allocation_math.h"
namespace {
using namespace tide_weighted_device;
using namespace tide::device_online::weighted_allocation;
__aicore__ inline int backward(__gm__ float* slopes,__gm__ float* g,I n,I width,I branch,
    I has_shift,__gm__ float* vectors,__gm__ float* scalars,__gm__ uint8_t* connected){
  if(branch==Absent||branch==Zero)return 0;
  if(branch<Caps||branch>Equality)return Domain;
  for(I j=0;j<n;++j)if(!finite(g[j]))return GradientOverflow;
  connected[Capacities]=1;
  if(branch==Caps){for(I j=0;j<n;++j)vectors[width+j]=g[j];return 0;}
  connected[Scores]=1;connected[Temp]=1;connected[ScoreShift]=has_shift;
  const I target=branch==Lower||branch==Equality?LowerBound:UpperBound;connected[target]=1;
  float total=0,largest=-1,anchor=0,offset=0;
  for(I j=0;j<n;++j){const float a=slopes[j*SlopeFields+Score];total+=a;
    if(a>largest){largest=a;anchor=g[j];}}
  if(!finite(total)||total<0)return GradientOverflow;
  const bool constraint=branch!=Slack&&total>0;
  if(constraint){for(I j=0;j<n;++j)offset+=slopes[j*SlopeFields+Score]*(g[j]-anchor);offset/=total;}
  const float mean=constraint?anchor+offset:0;float temp=0,shift=0;
  for(I j=0;j<n;++j){const float centered=constraint?(g[j]-anchor)-offset:g[j];
    const float ds=slopes[j*SlopeFields+Score]*centered,dc=slopes[j*SlopeFields+Cap]*centered;
    const float dt=slopes[j*SlopeFields+Tau]*centered;
    if(!finite(ds)||!finite(dc)||!finite(dt))return GradientOverflow;
    vectors[j]=ds;vectors[width+j]=dc;temp+=dt;shift-=ds;
  }
  scalars[target==LowerBound?Lo:Hi]=mean;scalars[Temperature]=temp;
  scalars[Shift]=has_shift&&branch==Slack?shift:0;
  for(I j=0;j<BoundFields;++j)if(!finite(scalars[j]))return GradientOverflow;
  return 0;
}
}
extern "C" __global__ __aicore__ void tide_weighted_allocation_vjp(GM_ADDR lengths,GM_ADDR branches,
    GM_ADDR slopes,GM_ADDR cotangent,GM_ADDR roots,GM_ADDR vectors,GM_ADDR scalars,GM_ADDR connections,
    GM_ADDR row_errors,GM_ADDR error,int64_t rows,int64_t width,int64_t has_shift){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)lengths);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  for(I row=AscendC::GetBlockIdx();row<rows;row+=AscendC::GetBlockNum()){
    auto v=(__gm__ float*)vectors+row*2*width,s=(__gm__ float*)scalars+row*BoundFields;
    auto con=(__gm__ uint8_t*)connections+row*Connections;auto e=(__gm__ int32_t*)row_errors+row;
    for(I j=0;j<2*width;++j)v[j]=0;for(I j=0;j<BoundFields;++j)s[j]=0;
    for(I j=0;j<Connections;++j)con[j]=0;*e=0;
    const I n=((__gm__ I*)lengths)[row];if(n<0||n>width){*e=2;continue;}
    if(!((__gm__ uint8_t*)roots)[row])continue;
    *e=backward((__gm__ float*)slopes+row*width*SlopeFields,(__gm__ float*)cotangent+row*width,
      n,width,((__gm__ I*)branches)[row],has_shift,v,s,con);
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
