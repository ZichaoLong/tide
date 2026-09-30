#include "fiber_vector.h"
extern "C" __global__ __aicore__ void tide_extra_full_reduce(GM_ADDR owners,GM_ADDR owner_count,GM_ADDR parameters,
    GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR oa,GM_ADDR ob,GM_ADDR oc,GM_ADDR error,int64_t na,int64_t nb,int64_t nc,int64_t chunk) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)owners);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  const I count=((__gm__ I*)owner_count)[0],ta=(na+255)/256,tb=(nb+255)/256,tc=(nc+255)/256,tiles=ta+tb+tc;
  tide_device::FiberVector v;v.init();auto total=v.x(),part=v.y();
  for(I task=AscendC::GetBlockIdx();task<count*tiles;task+=AscendC::GetBlockNum()) {
    const I node=((__gm__ I*)owners)[task/tiles],tile=task%tiles;
    const I extent=tile<ta?na:tile<ta+tb?nb:nc,start=(tile<ta?tile:tile<ta+tb?tile-ta:tile-ta-tb)*256;
    const uint32_t size=extent-start<256?extent-start:256;
    auto input=(__gm__ float*)(tile<ta?a:tile<ta+tb?b:c),output=(__gm__ float*)(tile<ta?oa:tile<ta+tb?ob:oc);
    v.load(total,output,node*extent+start,size);
    for(I row=0;row<chunk;++row)if(((__gm__ I*)parameters)[row]==node) {
      v.load(part,input,row*extent+start,size);AscendC::Add(total,total,part,size);AscendC::PipeBarrier<PIPE_V>();}
    v.save(total,output,node*extent+start,size);
  }
}
