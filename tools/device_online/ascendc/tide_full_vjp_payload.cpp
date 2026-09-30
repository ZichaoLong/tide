#include "fiber_vector.h"
namespace {using I=int64_t;}
// Copy only connected cotangents, including identity Full. There is no
// arithmetic on absent/poisoned values followed by a zero mask.
extern "C" __global__ __aicore__ void tide_full_vjp_payload(GM_ADDR values,GM_ADDR gradient,GM_ADDR connected,GM_ADDR comparison_connected,GM_ADDR count,
    GM_ADDR content,GM_ADDR comparison,GM_ADDR error,int64_t width) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)count);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  const I rows=((__gm__ I*)count)[0],tiles=(width+255)/256;
  tide_device::FiberVector vector;vector.init();auto x=vector.x();
  for(I task=AscendC::GetBlockIdx();task<rows*tiles;task+=AscendC::GetBlockNum()) {
    const I row=task/tiles,start=(task%tiles)*256;
    if(!((__gm__ uint8_t*)connected)[row])continue;
    const uint32_t size=width-start<256?width-start:256;
    vector.load(x,(__gm__ float*)gradient,row*width+start,size);vector.save(x,(__gm__ float*)content,row*width+start,size);
    if(((__gm__ uint8_t*)comparison_connected)[row]) {
      vector.load(x,(__gm__ float*)values,row*(5*width+2)+3*width+start,size);
      vector.save(x,(__gm__ float*)comparison,row*width+start,size);
    }
  }
}
