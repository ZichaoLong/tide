#include "fiber_vector.h"
extern "C" __global__ __aicore__ void tide_control_payload(GM_ADDR metadata,GM_ADDR values,GM_ADDR raw_full,GM_ADDR reads,
    GM_ADDR count,GM_ADDR range,GM_ADDR config,GM_ADDR gradient,GM_ADDR connected,GM_ADDR read_connected,GM_ADDR scores,
    GM_ADDR fresh,GM_ADDR cotangents,GM_ADDR products,GM_ADDR read_partials,GM_ADDR error,
    int64_t capacity,int64_t width,int64_t mode,int64_t emit_mode,int64_t fp16) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)metadata);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  const I rows=((__gm__ I*)count)[0],first=((__gm__ I*)range)[0],tiles=(width+255)/256,stride=5*width+2;
  auto e=(__gm__ I*)metadata,cfg=(__gm__ I*)config;auto v=(__gm__ float*)values,cot=(__gm__ float*)cotangents;
  tide_device::FiberVector vector;vector.init();auto x=vector.x(),y=vector.y();
  for(I task=AscendC::GetBlockIdx();task<rows*tiles;task+=AscendC::GetBlockNum()) {
    const I row=task/tiles,start=(task%tiles)*256,event=first+row,node=e[event*13+1],read=cfg[node*3+1];
    const uint32_t size=width-start<256?width-start:256;
    if(mode==0) {
      if(!((__gm__ uint8_t*)connected)[row])continue;
      vector.load(x,(__gm__ float*)gradient,row*width+start,size);
      const float probability=fp16?float(half(v[event*stride+5*width+1])):v[event*stride+5*width+1];
      if(emit_mode==2&&read>=0){AscendC::Muls(x,x,probability,size);AscendC::PipeBarrier<PIPE_V>();}
      vector.save(x,(__gm__ float*)fresh,row*width+start,size);
      if(read<0)continue;
      if(emit_mode==2) {
        vector.load(y,(__gm__ float*)gradient,row*width+start,size);AscendC::Sub(y,y,x,size);AscendC::PipeBarrier<PIPE_V>();
        vector.save(y,cot,row*5*width+start,size);
      }
      vector.load(x,(__gm__ float*)raw_full,event*width+start,size);vector.load(y,v,event*stride+start,size);
      AscendC::Sub(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();
      if(fp16) {
        // Both HST's declared saved delta and SOFTP's multiply consume the
        // actual half subtraction, while cotangents never round to half.
        auto rounded=y.ReinterpretCast<half>();
        AscendC::Cast(rounded,x,AscendC::RoundMode::CAST_RINT,size);AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(x,rounded,AscendC::RoundMode::CAST_NONE,size);AscendC::PipeBarrier<PIPE_V>();
      }
      vector.load(y,(__gm__ float*)gradient,row*width+start,size);AscendC::Mul(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();
      vector.save(x,(__gm__ float*)products,row*width+start,size);
    } else if(((__gm__ uint8_t*)read_connected)[row]) {
      const float score_gradient=((__gm__ float*)scores)[row];
      if(cfg[node*3+2]) {
        const float norm=v[event*stride+5*width];
        if(norm==0.f){AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
        else {vector.load(x,v,event*stride+read*width+start,size);AscendC::Duplicate(y,norm,size);AscendC::PipeBarrier<PIPE_V>();
          AscendC::Div(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();
          AscendC::Muls(x,x,score_gradient,size);AscendC::PipeBarrier<PIPE_V>();}
      } else {
        vector.load(x,v,event*stride+read*width+start,size);AscendC::Muls(x,x,score_gradient,size);AscendC::PipeBarrier<PIPE_V>();
        vector.save(x,(__gm__ float*)read_partials,row*width+start,size);
        if(fp16)vector.load(x,(__gm__ half*)reads,node*width+start,size);
        else vector.load(x,(__gm__ float*)reads,node*width+start,size);
        AscendC::Muls(x,x,score_gradient,size);AscendC::PipeBarrier<PIPE_V>();
      }
      vector.load(y,cot,(row*5+read)*width+start,size);AscendC::Add(y,y,x,size);AscendC::PipeBarrier<PIPE_V>();
      vector.save(y,cot,(row*5+read)*width+start,size);
    }
  }
}
