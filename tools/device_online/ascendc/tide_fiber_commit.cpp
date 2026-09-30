#include "fiber_vector.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_fiber_commit(GM_ADDR events,GM_ADDR counts,GM_ADDR config,
    GM_ADDR active,GM_ADDR key,GM_ADDR value,GM_ADDR bias,GM_ADDR live_key,GM_ADDR live_value,
    GM_ADDR live_bias,GM_ADDR lengths,GM_ADDR peak,GM_ADDR error,int64_t width,int64_t capacity) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)events);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto e=(__gm__ I*)events,c=(__gm__ I*)counts,cfg=(__gm__ I*)config;
  auto on=(__gm__ uint8_t*)active;auto n=(__gm__ I*)lengths,maximum=(__gm__ I*)peak;
  tide_device::FiberVector op;op.init();auto x=op.x();const I tiles=(width+255)/256;
  if(AscendC::GetBlockIdx()==0)for(I i=0;i<c[1];++i) {
    const I parameter=e[i*7+2],fiber=e[i*7],owner=e[i*7+1];
    if(e[i*7+4]>maximum[0])maximum[0]=e[i*7+4];
    if(!(cfg[parameter*2]||on[fiber]))continue;
    const I length=cfg[parameter*2+1]&&on[fiber]?0:e[i*7+4];n[owner]=length;
    for(I row=0;row<length;++row)((__gm__ float*)live_bias)[owner*capacity+row]=((__gm__ float*)bias)[owner*capacity+row];
  }
  for(I task=AscendC::GetBlockIdx();task<c[1]*tiles;task+=AscendC::GetBlockNum()) {
    const I event=task/tiles,start=(task%tiles)*256,parameter=e[event*7+2],owner=e[event*7+1],fiber=e[event*7];
    const bool adopt=cfg[parameter*2]||on[fiber],clear=cfg[parameter*2+1]&&on[fiber];
    if(!adopt)continue;
    const I length=clear?0:e[event*7+4];
    const uint32_t size=width-start<256?width-start:256;
    for(I row=0;row<length;++row) {
      const I src=owner*capacity+row;
      op.load(x,(__gm__ float*)key,src*width+start,size);op.save(x,(__gm__ float*)live_key,src*width+start,size);
      op.load(x,(__gm__ float*)value,src*width+start,size);op.save(x,(__gm__ float*)live_value,src*width+start,size);
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
