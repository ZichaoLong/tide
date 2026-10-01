#include "fiber_vector.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_event_reverse_reduce(GM_ADDR plan,GM_ADDR flags,GM_ADDR events,
    GM_ADDR node_offsets,GM_ADDR input_gradient,GM_ADDR qkv_gradient,GM_ADDR projection_gradient,
    GM_ADDR content,GM_ADDR content_connected,GM_ADDR parameters,GM_ADDR parameter_connected,GM_ADDR error,
    int64_t chunk,int64_t width,int64_t kv_width,int64_t nodes) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)plan);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto rows=(__gm__ I*)plan,e=(__gm__ I*)events,offsets=(__gm__ I*)node_offsets;auto f=(__gm__ uint8_t*)flags;
  const I cols=width+2*kv_width,tiles=(width+255)/256;
  if(AscendC::GetBlockIdx()==0)for(I i=0;i<chunk;++i)if(rows[i*8]>=0) {
    const I n=e[rows[i*8]*13+1],local=rows[i*8+3];
    ((__gm__ uint8_t*)content_connected)[local]|=f[i*6]||f[i*6+1]||f[i*6+2];
    for(I j=0;j<4;++j)((__gm__ uint8_t*)parameter_connected)[n*4+j]|=f[i*6+j];
  }
  tide_device::FiberVector op;op.init();auto x=op.x(),y=op.y();
  for(I task=AscendC::GetBlockIdx();task<chunk*tiles;task+=AscendC::GetBlockNum()) {
    const I i=task/tiles,start=(task%tiles)*256;if(rows[i*8]<0)continue;
    const uint32_t size=width-start<256?width-start:256;const I dst=rows[i*8+3]*width+start;
    op.load(x,(__gm__ float*)content,dst,size);op.load(y,(__gm__ float*)input_gradient,i*width+start,size);
    AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();op.save(x,(__gm__ float*)content,dst,size);
  }
  // Each node/parameter-feature tile has one writer; repeated samples are
  // reduced in the stable batch order, without conflicting scatter updates.
  for(I task=AscendC::GetBlockIdx();task<nodes*width*4*tiles;task+=AscendC::GetBlockNum()) {
    const I start=(task%tiles)*256,kind=(task/tiles)%4,row=(task/tiles/4)%width,n=task/tiles/4/width;
    if(offsets[n]==offsets[n+1])continue;
    const I length=kind==0||kind==3?width:kv_width;if(start>=length)continue;
    const uint32_t size=length-start<256?length-start:256;
    const I base=offsets[n]+(kind==0?0:kind==1?width*width:kind==2?width*(width+kv_width):width*cols);
    const I dst=base+row*length+start;bool loaded=false;
    for(I i=0;i<chunk;++i)if(rows[i*8]>=0&&e[rows[i*8]*13+1]==n&&f[i*6+kind]) {
      if(!loaded){op.load(x,(__gm__ float*)parameters,dst,size);loaded=true;}
      const I col=kind==0?0:kind==1?width:width+kv_width;
      op.load(y,(__gm__ float*)(kind==3?projection_gradient:qkv_gradient),
        kind==3?(i*width+row)*width+start:(i*width+row)*cols+col+start,size);
      AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();
    }
    if(loaded)op.save(x,(__gm__ float*)parameters,dst,size);
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
