#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_fiber_reverse_plan(GM_ADDR events,GM_ADDR config,
    GM_ADDR ranges,GM_ADDR previous,GM_ADDR tails,GM_ADDR stage,GM_ADDR proposals,
    GM_ADDR key_connected,GM_ADDR value_connected,GM_ADDR bias_connected,
    GM_ADDR plan,GM_ADDR flags,GM_ADDR branch,GM_ADDR error,int64_t owners,int64_t parameters,int64_t chunk) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)events);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error,go=(__gm__ int32_t*)branch;go[0]=0;
  auto r=(__gm__ I*)ranges,prev=(__gm__ I*)previous,tail=(__gm__ I*)tails,interval=(__gm__ I*)stage,p=(__gm__ I*)plan;
  auto f=(__gm__ uint8_t*)flags,kc=(__gm__ uint8_t*)key_connected,vc=(__gm__ uint8_t*)value_connected,bc=(__gm__ uint8_t*)bias_connected;
  for(I i=0;i<chunk;++i){for(I j=0;j<10;++j)p[i*10+j]=-1;for(I j=0;j<4;++j)f[i*4+j]=0;}
  if(status[0])return;I count=0;
  for(I owner=0;owner<owners&&count<chunk;++owner) {
    const I event=tail[owner];if(event<interval[0])continue;
    if(event>=interval[1]){status[0]=2;break;}
    auto row=p+count*10;row[0]=event;row[1]=owner;row[2]=owner%parameters;row[3]=event-interval[0];
    for(I j=0;j<6;++j)row[4+j]=r[event*6+j];
    f[count*4]=((__gm__ uint8_t*)proposals)[row[3]];f[count*4+1]=kc[owner];f[count*4+2]=vc[owner];f[count*4+3]=bc[owner];
    tail[owner]=prev[event];++count;
  }
  if(!status[0]&&count)go[0]=1;
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
