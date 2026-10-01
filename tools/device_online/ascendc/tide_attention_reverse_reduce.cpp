#include "fiber_vector.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_attention_reverse_reduce(GM_ADDR key_partial,GM_ADDR value_partial,
    GM_ADDR score_gradient,GM_ADDR valid,GM_ADDR cursor,GM_ADDR key_gradient,GM_ADDR value_gradient,
    GM_ADDR bias_gradient,GM_ADDR error,int64_t queries,int64_t heads,int64_t kv_heads,
    int64_t width,int64_t capacity,int64_t tile) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)valid);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  tide_device::FiberVector op;op.init();auto x=op.x(),y=op.y();
  const I features=(width+255)/256,first=((__gm__ I*)cursor)[1],groups=heads/kv_heads;
  for(I task=AscendC::GetBlockIdx();task<queries*kv_heads*tile*features;task+=AscendC::GetBlockNum()) {
    const I feature=(task%features)*256,row=task/features,k=row%tile,h=(row/tile)%kv_heads,q=row/(tile*kv_heads);
    if(k>=((__gm__ I*)valid)[q])continue;
    const uint32_t size=width-feature<256?width-feature:256;
    for(I part=0;part<2;++part) {
      auto src=(__gm__ float*)(part?value_partial:key_partial),dst=(__gm__ float*)(part?value_gradient:key_gradient);
      AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();
      for(I g=0;g<groups;++g) {
        op.load(y,src,((q*heads+h*groups+g)*tile+k)*width+feature,size);
        AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();
      }
      op.save(x,dst,((q*kv_heads+h)*capacity+first+k)*width+feature,size);
    }
  }
  if(AscendC::GetBlockIdx()==0)for(I q=0;q<queries;++q)for(I k=0;k<((__gm__ I*)valid)[q];++k) {
    float sum=0;for(I h=0;h<heads;++h)sum+=((__gm__ float*)score_gradient)[(q*heads+h)*tile+k];
    ((__gm__ float*)bias_gradient)[q*capacity+first+k]=sum;
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
