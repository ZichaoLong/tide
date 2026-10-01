#include "fiber_vector.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_attention_reverse_weights(GM_ADDR logits,GM_ADDR valid,
    GM_ADDR normalization,GM_ADDR probability_gradient,GM_ADDR correction,GM_ADDR probability,
    GM_ADDR score_gradient,GM_ADDR error,int64_t rows,int64_t heads,int64_t tile) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)valid);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  tide_device::FiberVector op;op.init();auto x=op.x(),y=op.y();
  for(I row=AscendC::GetBlockIdx();row<rows*heads;row+=AscendC::GetBlockNum()) {
    const uint32_t size=((__gm__ I*)valid)[row/heads];
    AscendC::Duplicate(x,0.f,uint32_t(tile));AscendC::Duplicate(y,0.f,uint32_t(tile));AscendC::PipeBarrier<PIPE_V>();
    if(size) {
      const float maximum=((__gm__ float*)normalization)[row*8],denominator=((__gm__ float*)normalization)[row*8+1];
      op.load(x,(__gm__ float*)logits,row*tile,size);
      AscendC::Adds(x,x,-maximum,size);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Exp(x,x,size);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Muls(x,x,1.f/denominator,size);AscendC::PipeBarrier<PIPE_V>();
      op.load(y,(__gm__ float*)probability_gradient,row*tile,size);
      AscendC::Adds(y,y,-((__gm__ float*)correction)[row],size);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(y,y,x,size);AscendC::PipeBarrier<PIPE_V>();
    }
    op.save(x,(__gm__ float*)probability,row*tile,uint32_t(tile));
    op.save(y,(__gm__ float*)score_gradient,row*tile,uint32_t(tile));
  }
}
