#include "fiber_vector.h"
namespace {using I=int64_t;
// mode0: decay real old biases, mode1: QKV placement, mode2: pool query rows.
template<class T>
__aicore__ inline void run(GM_ADDR events,GM_ADDR tokens,GM_ADDR counts,
    GM_ADDR ids,GM_ADDR heads,GM_ADDR scales,GM_ADDR projection,GM_ADDR queries,GM_ADDR key,GM_ADDR value,
    GM_ADDR bias,GM_ADDR query_bias,GM_ADDR decay,GM_ADDR pool_kinds,GM_ADDR coefficients,GM_ADDR pooled,GM_ADDR error,
    int64_t width,int64_t capacity,int64_t chunk,int64_t mode) {
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)events);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto e=(__gm__ I*)events,t=(__gm__ I*)tokens,c=(__gm__ I*)counts,index=(__gm__ I*)ids;
  tide_device::FiberVector op;op.init();auto x=op.x(),y=op.y();
  const I tiles=(width+255)/256,cache_tiles=(capacity+255)/256;
  const I tasks=mode==0?c[1]*cache_tiles:(mode==1?chunk:c[1])*tiles;
  for(I task=AscendC::GetBlockIdx();task<tasks;task+=AscendC::GetBlockNum()) {
    if(mode==0) {
      const I first=task/cache_tiles,start=(task%cache_tiles)*256,owner=e[first*7+1],parameter=e[first*7+2];
      if(first&&e[(first-1)*7+1]==owner)continue;
      // Only the existing prefix is decayed. The untouched tail stays zero
      // until a later event appends its keys; no per-message host placement.
      AscendC::Duplicate(x,0.f,256);AscendC::PipeBarrier<PIPE_V>();
      const I initial=e[first*7+3];
      const uint32_t initial_size=start<initial?(initial-start<256?initial-start:256):0;
      if(initial_size)op.load(x,(__gm__ T*)bias,owner*capacity+start,initial_size);
      const float rate=float(((__gm__ T*)decay)[parameter]);
      for(I event=first;event<c[1]&&e[event*7+1]==owner;++event) {
        const I old=e[event*7+3],length=e[event*7+4];if(start>=length)continue;
        const uint32_t size=length-start<256?length-start:256,old_size=start<old?(old-start<256?old-start:256):0;
        if(old_size)for(I tick=0;tick<e[event*7+6];++tick){
          AscendC::Adds(x,x,-rate,old_size);AscendC::PipeBarrier<PIPE_V>();
          // A node-time batch preserves each declared half-precision tick.
          if constexpr(sizeof(T)==2) {
            auto rounded=y.ReinterpretCast<half>();
            AscendC::Cast(rounded,x,AscendC::RoundMode::CAST_RINT,old_size);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(x,rounded,AscendC::RoundMode::CAST_NONE,old_size);AscendC::PipeBarrier<PIPE_V>();
          }
        }
        op.save(x,(__gm__ T*)query_bias,event*capacity+start,size);
        if(event+1==c[1]||e[(event+1)*7+1]!=owner)op.save(x,(__gm__ T*)bias,owner*capacity+start,size);
      }
    }else if(mode==1) {
      const I row=task/tiles,start=(task%tiles)*256,token=index[row];if(token<0)continue;
      const I event=t[token*4+1],owner=e[event*7+1],position=t[token*4+2],parameter=t[token*4+3];
      const uint32_t size=width-start<256?width-start:256;
      op.load(x,(__gm__ T*)projection,row*3*width+start,size);
      const float factor=((__gm__ float*)scales)[parameter];
      AscendC::Muls(x,x,factor,size);AscendC::PipeBarrier<PIPE_V>();
      op.save(x,(__gm__ T*)queries,token*width+start,size);
      op.load(x,(__gm__ T*)projection,row*3*width+width+start,size);
      op.save(x,(__gm__ T*)key,(owner*capacity+position)*width+start,size);
      op.load(x,(__gm__ T*)projection,row*3*width+2*width+start,size);
      op.save(x,(__gm__ T*)value,(owner*capacity+position)*width+start,size);
    }else {
      const I event=task/tiles,start=(task%tiles)*256,first=e[event*7+5],count=e[event*7+4]-e[event*7+3];
      const I kind=((__gm__ I*)pool_kinds)[e[event*7+2]];
      const uint32_t size=width-start<256?width-start:256;
      AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();
      for(I row=first;row<first+count;++row){op.load(y,(__gm__ T*)queries,row*width+start,size);
        if(kind>=2){const float coefficient=((__gm__ float*)coefficients)[row];
          AscendC::Muls(y,y,coefficient,size);AscendC::PipeBarrier<PIPE_V>();}
        AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();}
      if(kind==1){AscendC::Muls(x,x,1.f/static_cast<float>(count),size);AscendC::PipeBarrier<PIPE_V>();}
      op.save(x,(__gm__ T*)pooled,e[event*7]*width+start,size);
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
} // namespace
extern "C" __global__ __aicore__ void tide_fiber_payload(GM_ADDR events,GM_ADDR tokens,GM_ADDR counts,
    GM_ADDR ids,GM_ADDR heads,GM_ADDR scales,GM_ADDR projection,GM_ADDR queries,GM_ADDR key,GM_ADDR value,
    GM_ADDR bias,GM_ADDR query_bias,GM_ADDR decay,GM_ADDR pool_kinds,GM_ADDR coefficients,GM_ADDR pooled,GM_ADDR error,
    int64_t width,int64_t capacity,int64_t chunk,int64_t mode,int64_t fp16) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(fp16)run<half>(events,tokens,counts,ids,heads,scales,projection,queries,key,value,bias,query_bias,decay,pool_kinds,coefficients,pooled,error,width,capacity,chunk,mode);
  else run<float>(events,tokens,counts,ids,heads,scales,projection,queries,key,value,bias,query_bias,decay,pool_kinds,coefficients,pooled,error,width,capacity,chunk,mode);
}
