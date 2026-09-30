#include "fiber_vector.h"
extern "C" __global__ __aicore__ void tide_aggregate_vjp_reduce(GM_ADDR row_nodes,GM_ADDR owners,GM_ADDR owner_count,
    GM_ADDR partials,GM_ADDR output,GM_ADDR error,int64_t chunk,int64_t slots) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)owners);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  const I count=((__gm__ I*)owner_count)[0],tiles=(slots+255)/256;
  tide_device::FiberVector v;v.init();auto total=v.x(),part=v.y();
  for(I task=AscendC::GetBlockIdx();task<count*tiles;task+=AscendC::GetBlockNum()) {
    const I owner=((__gm__ I*)owners)[task/tiles],start=(task%tiles)*256;
    const uint32_t size=slots-start<256?slots-start:256;
    v.load(total,(__gm__ float*)output,owner*slots+start,size);
    for(I row=0;row<chunk;++row)if(((__gm__ I*)row_nodes)[row]==owner) {
      v.load(part,(__gm__ float*)partials,row*slots+start,size);AscendC::Add(total,total,part,size);AscendC::PipeBarrier<PIPE_V>();}
    v.save(total,(__gm__ float*)output,owner*slots+start,size);
  }
}
