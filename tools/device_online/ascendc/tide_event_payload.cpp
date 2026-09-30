#include "fiber_vector.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_event_payload(GM_ADDR events,GM_ADDR ids,GM_ADDR projected,
    GM_ADDR queries,GM_ADDR key,GM_ADDR value,GM_ADDR error,
    int64_t width,int64_t kv_width,int64_t capacity,int64_t owners,int64_t chunk) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)events);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto e=(__gm__ I*)events,index=(__gm__ I*)ids;
  tide_device::FiberVector op;op.init();auto x=op.x();const I tiles=(width+255)/256;
  for(I task=AscendC::GetBlockIdx();task<chunk*tiles;task+=AscendC::GetBlockNum()) {
    const I row=task/tiles,start=(task%tiles)*256,event=index[row];
    const uint32_t size=width-start<256?width-start:256;
    if(event<0)continue;
    op.load(x,(__gm__ float*)projected,row*(width+2*kv_width)+start,size);
    op.save(x,(__gm__ float*)queries,e[event*7]*width+start,size);
    if(start>=kv_width)continue;
    const uint32_t kv_size=kv_width-start<256?kv_width-start:256;const I target=owners*capacity+event;
    op.load(x,(__gm__ float*)projected,row*(width+2*kv_width)+width+start,kv_size);
    op.save(x,(__gm__ float*)key,target*kv_width+start,kv_size);
    op.load(x,(__gm__ float*)projected,row*(width+2*kv_width)+width+kv_width+start,kv_size);
    op.save(x,(__gm__ float*)value,target*kv_width+start,kv_size);
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
