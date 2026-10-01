#include "fiber_vector.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_attention_reverse_pack(GM_ADDR query,GM_ADDR key,GM_ADDR value,
    GM_ADDR cotangent,GM_ADDR bias,GM_ADDR connected,GM_ADDR cursor,GM_ADDR valid,
    GM_ADDR safe_query,GM_ADDR safe_cotangent,GM_ADDR packed_key,GM_ADDR packed_value,GM_ADDR additive,
    GM_ADDR error,int64_t queries,int64_t heads,int64_t kv_heads,int64_t width,int64_t capacity,int64_t tile,int64_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)cursor);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  tide_device::FiberVector op;op.init();auto x=op.x();const I features=(width+255)/256;
  auto on=(__gm__ uint8_t*)connected;const I start=((__gm__ I*)cursor)[1];
  const I rows=queries*heads*(mode?tile:1);
  for(I task=AscendC::GetBlockIdx();task<rows*features;task+=AscendC::GetBlockNum()) {
    const I row=task/features,feature=(task%features)*256;
    const I q=mode?row/(heads*tile):row/heads,h=mode?(row/tile)%heads:row%heads,k=mode?row%tile:0;
    const uint32_t size=width-feature<256?width-feature:256;
    const bool present=on[q]&&(!mode||k<((__gm__ I*)valid)[q]);
    for(I part=0;part<2;++part) {
      auto src=(__gm__ float*)(mode?(part?value:key):(part?cotangent:query));
      auto dst=(__gm__ float*)(mode?(part?packed_value:packed_key):(part?safe_cotangent:safe_query));
      const I at=mode?((q*kv_heads+h/(heads/kv_heads))*capacity+start+k)*width+feature:row*width+feature;
      if(present)op.load(x,src,at,size);else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
      op.save(x,dst,row*width+feature,size);
    }
  }
  // One writer owns adjacent scalar bias/mask words; head broadcasting happens
  // in the numerical operator, not through racing small GM stores.
  if(mode&&AscendC::GetBlockIdx()==0)for(I q=0;q<queries;++q)for(I k=0;k<tile;++k)
    ((__gm__ float*)additive)[q*tile+k]=k<((__gm__ I*)valid)[q]?((__gm__ float*)bias)[q*capacity+start+k]:0.f;
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
