#include "fiber_vector.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_fiber_vjp_sources(GM_ADDR plan,GM_ADDR row_gradient,
    GM_ADDR weight_gradient,GM_ADDR bias_gradient,GM_ADDR rows,GM_ADDR weights,GM_ADDR biases,
    GM_ADDR error,int64_t batch,int64_t sources,int64_t width,int64_t chunk) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)plan);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto ids=(__gm__ I*)plan;const I wt=(width+255)/256,cols=3*width,tiles=(cols+255)/256;
  tide_device::FiberVector op;op.init();auto x=op.x(),y=op.y();
  for(I task=AscendC::GetBlockIdx();task<chunk*wt;task+=AscendC::GetBlockNum()) {
    const I i=task/wt,start=(task%wt)*256,b=ids[i*3],r=ids[i*3+1];if(b<0)continue;
    const uint32_t size=width-start<256?width-start:256;
    op.load(x,(__gm__ float*)row_gradient,i*width+start,size);
    op.save(x,(__gm__ float*)rows,(b*sources+r)*width+start,size);
  }
  for(I task=AscendC::GetBlockIdx();task<batch*(width+1)*tiles;task+=AscendC::GetBlockNum()) {
    const I b=task/(width+1)/tiles,r=(task/tiles)%(width+1),start=(task%tiles)*256;
    const uint32_t size=cols-start<256?cols-start:256;
    auto to=(__gm__ float*)(r==width?biases:weights),from=(__gm__ float*)(r==width?bias_gradient:weight_gradient);
    const I at=(r==width?b:b*width+r)*cols+start;
    op.load(x,to,at,size);
    for(I i=0;i<chunk;++i)if(ids[i*3]==b) {
      op.load(y,from,(r==width?i:i*width+r)*cols+start,size);
      AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();
    }
    op.save(x,to,at,size);
  }
}
