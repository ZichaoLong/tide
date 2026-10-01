#include "fiber_vector.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_fiber_vjp_queries(GM_ADDR plan,GM_ADDR coefficients,
    GM_ADDR output,GM_ADDR query,GM_ADDR key,GM_ADDR value,GM_ADDR bias,GM_ADDR pool_partial,
    GM_ADDR pooled,GM_ADDR source_gradient,GM_ADDR cache_key,GM_ADDR cache_value,GM_ADDR cache_bias,
    GM_ADDR pool_gradient,GM_ADDR error,int64_t batch,int64_t sources,int64_t width,int64_t heads,
    int64_t capacity,int64_t domain,int64_t chunk) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)plan);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto ids=(__gm__ I*)plan;const I d=width/heads,wt=(width+255)/256,dt=(d+255)/256;
  tide_device::FiberVector op;op.init();auto x=op.x(),y=op.y();
  for(I task=AscendC::GetBlockIdx();task<chunk*wt;task+=AscendC::GetBlockNum()) {
    const I i=task/wt,start=(task%wt)*256,b=ids[i*3],r=ids[i*3+1];if(b<0)continue;
    const uint32_t size=width-start<256?width-start:256;
    op.load(x,(__gm__ float*)query,i*width+start,size);
    op.save(x,(__gm__ float*)source_gradient,(b*sources+r)*width+start,size);
  }
  for(I task=AscendC::GetBlockIdx();task<batch*wt;task+=AscendC::GetBlockNum()) {
    const I b=task/wt,start=(task%wt)*256;const uint32_t size=width-start<256?width-start:256;
    op.load(x,(__gm__ float*)pooled,b*width+start,size);
    for(I i=0;i<chunk;++i)if(ids[i*3]==b) {
      op.load(y,(__gm__ float*)output,i*width+start,size);
      const float coefficient=((__gm__ float*)coefficients)[b*sources+ids[i*3+1]];
      AscendC::Muls(y,y,coefficient,size);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();
    }
    op.save(x,(__gm__ float*)pooled,b*width+start,size);
  }
  for(I task=AscendC::GetBlockIdx();task<batch*heads*capacity*dt;task+=AscendC::GetBlockNum()) {
    const I start=(task%dt)*256,k=(task/dt)%capacity,h=(task/dt/capacity)%heads,b=task/dt/capacity/heads;
    const I at=((b*heads+h)*capacity+k)*d+start;const uint32_t size=d-start<256?d-start:256;
    for(I which=0;which<2;++which) {
      auto to=(__gm__ float*)(which?cache_value:cache_key),from=(__gm__ float*)(which?value:key);
      op.load(x,to,at,size);
      for(I i=0;i<chunk;++i)if(ids[i*3]==b) {
        op.load(y,from,((i*heads+h)*capacity+k)*d+start,size);
        AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();
      }
      op.save(x,to,at,size);
    }
  }
  // Scalar metadata-shaped reductions still have unique owners and stable
  // query order. No concurrent scatter-add targets the same cache/slot.
  if(AscendC::GetBlockIdx()==0)for(I task=0;task<batch*(capacity+domain);++task) {
    const I b=task/(capacity+domain),j=task%(capacity+domain);
    if(j<capacity) {
      float total=((__gm__ float*)cache_bias)[b*capacity+j];
      for(I i=0;i<chunk;++i)if(ids[i*3]==b)total+=((__gm__ float*)bias)[i*capacity+j];
      ((__gm__ float*)cache_bias)[b*capacity+j]=total;
    }else {
      const I slot=j-capacity,at=b*domain+slot;float total=((__gm__ float*)pool_gradient)[at];
      for(I i=0;i<chunk;++i)if(ids[i*3]==b&&ids[i*3+2]==slot)total+=((__gm__ float*)pool_partial)[i];
      ((__gm__ float*)pool_gradient)[at]=total;
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
