#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_state_read_plan(GM_ADDR metadata,GM_ADDR count,GM_ADDR modes,GM_ADDR kinds,
    GM_ADDR connected,GM_ADDR head,GM_ADDR next,GM_ADDR parameters,GM_ADDR error,int64_t capacity,int64_t nodes) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)metadata);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;auto e=(__gm__ I*)metadata;auto h=(__gm__ I*)head,nx=(__gm__ I*)next;
  const I size=((__gm__ I*)count)[0];for(I n=0;n<nodes;++n)h[n]=-1;
  if(!status[0]&&(size<0||size>capacity))status[0]=2;
  for(I row=size;row>0&&!status[0];){--row;const I node=e[row*13+1];
    if(node<0||node>=nodes){status[0]=2;break;}nx[row]=h[node];h[node]=row;
    if(((__gm__ uint8_t*)connected)[row]) {
      const I mode=((__gm__ I*)modes)[node],kind=((__gm__ I*)kinds)[node];
      if(mode<0||mode>2||kind<0||kind>1){status[0]=2;break;}
      if(!kind)((__gm__ uint8_t*)parameters)[node]=1;
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
