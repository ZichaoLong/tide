#include "fiber_vector.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_event_reverse_pack(GM_ADDR plan,GM_ADDR flags,GM_ADDR events,
    GM_ADDR values,GM_ADDR proposals,GM_ADDR qkv,GM_ADDR projection,GM_ADDR journal,
    GM_ADDR content,GM_ADDR weights,GM_ADDR output_weights,GM_ADDR cotangent,GM_ADDR projected,
    GM_ADDR query,GM_ADDR key,GM_ADDR value,GM_ADDR connected,GM_ADDR error,
    int64_t chunk,int64_t width,int64_t heads,int64_t kv_heads,int64_t capacity,int64_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)plan);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto rows=(__gm__ I*)plan;auto f=(__gm__ uint8_t*)flags;
  const I d=width/heads,kv=d*kv_heads,cols=width+2*kv,tiles=(width+255)/256;
  tide_device::FiberVector op;op.init();auto x=op.x();
  if(mode==0) {
    if(AscendC::GetBlockIdx()==0)for(I i=0;i<chunk;++i)((__gm__ uint8_t*)connected)[i]=f[i*6];
    for(I task=AscendC::GetBlockIdx();task<chunk*tiles;task+=AscendC::GetBlockNum()) {
      const I i=task/tiles,start=(task%tiles)*256;const uint32_t size=width-start<256?width-start:256;
      if(f[i*6]||f[i*6+1]||f[i*6+2])op.load(x,(__gm__ float*)values,rows[i*8]*(5*width+2)+start,size);
      else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
      op.save(x,(__gm__ float*)content,i*width+start,size);
      if(f[i*6])op.load(x,(__gm__ float*)proposals,rows[i*8+3]*width+start,size);
      else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
      op.save(x,(__gm__ float*)cotangent,i*width+start,size);
    }
    // Gather parameter segments only when connected. A missing Q/O branch
    // must not evaluate unused large/nonfinite projection parameters.
    for(I task=AscendC::GetBlockIdx();task<chunk*width*4*tiles;task+=AscendC::GetBlockNum()) {
      const I start=(task%tiles)*256,segment=(task/tiles)%4,row=(task/tiles/4)%width,i=task/tiles/4/width;
      const I length=segment==0||segment==3?width:kv;if(start>=length)continue;
      const uint32_t size=length-start<256?length-start:256;const I param=rows[i*8+2];
      const I offset=segment==0?0:segment==1?width:width+kv;
      if(f[i*6+segment])op.load(x,(__gm__ float*)(segment==3?projection:qkv),
        segment==3?(param*width+row)*width+start:(param*width+row)*cols+offset+start,size);
      else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
      op.save(x,(__gm__ float*)(segment==3?output_weights:weights),
        segment==3?(i*width+row)*width+start:(i*width+row)*cols+offset+start,size);
    }
  } else {
    for(I task=AscendC::GetBlockIdx();task<chunk*tiles;task+=AscendC::GetBlockNum()) {
      const I i=task/tiles,start=(task%tiles)*256;const uint32_t size=width-start<256?width-start:256;
      if(f[i*6])op.load(x,(__gm__ float*)projected,i*cols+start,size);
      else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
      op.save(x,(__gm__ float*)query,i*width+start,size);
    }
    const I dt=(d+255)/256;
    for(I task=AscendC::GetBlockIdx();task<chunk*kv_heads*capacity*dt;task+=AscendC::GetBlockNum()) {
      const I start=(task%dt)*256,k=(task/dt)%capacity,h=(task/dt/capacity)%kv_heads,i=task/dt/capacity/kv_heads;
      const uint32_t size=d-start<256?d-start:256;
      for(I which=0;which<2;++which) {
        if(f[i*6]&&k<rows[i*8+7])op.load(x,(__gm__ float*)journal,(rows[i*8+6]+k)*2*kv+which*kv+h*d+start,size);
        else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
        op.save(x,(__gm__ float*)(which?value:key),((i*kv_heads+h)*capacity+k)*d+start,size);
      }
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
