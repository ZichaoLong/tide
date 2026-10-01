#include "fiber_vector.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_event_reverse_fold(GM_ADDR plan,GM_ADDR flags,GM_ADDR events,GM_ADDR config,
    GM_ADDR query,GM_ADDR key,GM_ADDR value,GM_ADDR carry_key,GM_ADDR carry_value,GM_ADDR projected,GM_ADDR error,
    int64_t chunk,int64_t width,int64_t heads,int64_t kv_heads,int64_t capacity) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)plan);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto rows=(__gm__ I*)plan,e=(__gm__ I*)events,cfg=(__gm__ I*)config;auto f=(__gm__ uint8_t*)flags;
  const I d=width/heads,kv=kv_heads*d,cols=width+2*kv,wt=(width+255)/256,dt=(d+255)/256;
  tide_device::FiberVector op;op.init();auto x=op.x(),y=op.y();
  for(I task=AscendC::GetBlockIdx();task<chunk*wt;task+=AscendC::GetBlockNum()) {
    const I i=task/wt,start=(task%wt)*256;const uint32_t size=width-start<256?width-start:256;
    op.load(x,(__gm__ float*)query,i*width+start,size);op.save(x,(__gm__ float*)projected,i*cols+start,size);
  }
  for(I task=AscendC::GetBlockIdx();task<chunk*kv_heads*dt;task+=AscendC::GetBlockNum()) {
    const I start=(task%dt)*256,h=(task/dt)%kv_heads,i=task/dt/kv_heads;
    const uint32_t size=d-start<256?d-start:256;const I event=rows[i*8],owner=rows[i*8+1],param=rows[i*8+2];
    const I old=event<0?0:rows[i*8+5],prop=event<0?0:rows[i*8+7],drop=old+1-prop;
    const bool active=event>=0&&e[event*13+3],adopt=event>=0&&(cfg[param*2]||active),clear=event>=0&&cfg[param*2+1]&&active;
    for(I which=0;which<2;++which) {
      auto incoming=(__gm__ float*)(which?value:key),carry=(__gm__ float*)(which?carry_value:carry_key);
      // Read the new observation's gradient before overwriting the carry arena.
      AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();
      if(event>=0&&f[i*6])op.load(x,incoming,((i*kv_heads+h)*capacity+prop-1)*d+start,size);
      if(adopt&&!clear&&f[i*6+4+which]) {
        op.load(y,carry,(owner*capacity+prop-1)*kv+h*d+start,size);
        AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();
      }
      op.save(x,(__gm__ float*)projected,i*cols+width+which*kv+h*d+start,size);
      if(event<0)continue;
      // Sliding-window indices move left. Descending writes avoid overwriting
      // a still-needed incoming carry; owners in a batch are distinct.
      for(I j=capacity;j>0;) {--j;
        AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();
        if(j<old) {
          if(!adopt&&f[i*6+4+which])op.load(x,carry,(owner*capacity+j)*kv+h*d+start,size);
          const I at=j-drop;
          if(at>=0&&at<prop-1) {
            if(f[i*6]) {op.load(y,incoming,((i*kv_heads+h)*capacity+at)*d+start,size);
              AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();}
            if(adopt&&!clear&&f[i*6+4+which]) {op.load(y,carry,(owner*capacity+at)*kv+h*d+start,size);
              AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();}
          }
        }
        op.save(x,carry,(owner*capacity+j)*kv+h*d+start,size);
      }
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
