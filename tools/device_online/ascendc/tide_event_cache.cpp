#include "fiber_vector.h"
#include "event_sequence.h"
namespace {using I=int64_t;}
// mode1 commits the final adopted cache/clear for each owner;
// mode2 writes optional old/proposed diagnostic rows, without committing state.
extern "C" __global__ __aicore__ void tide_event_cache(GM_ADDR events,GM_ADDR counts,GM_ADDR config,
    GM_ADDR active,GM_ADDR fibers,GM_ADDR key,GM_ADDR value,GM_ADDR live_key,GM_ADDR live_value,
    GM_ADDR lengths,GM_ADDR peak,GM_ADDR meta,GM_ADDR trace,GM_ADDR trace_count,GM_ADDR error,
    int64_t width,int64_t capacity,int64_t owners,int64_t trace_rows,int64_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)events);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;if(status[0])return;
  auto e=(__gm__ I*)events,c=(__gm__ I*)counts,cfg=(__gm__ I*)config,f=(__gm__ I*)fibers;
  auto on=(__gm__ uint8_t*)active;auto n=(__gm__ I*)lengths,maximum=(__gm__ I*)peak;
  if(mode==2) {
    if(AscendC::GetBlockIdx()!=0)return;
    I needed=0;auto size=(__gm__ I*)trace_count;size[0]=0;
    for(I event=0;event<c[1];++event)for(I kind=0;kind<2;++kind) {
      const I length=e[event*7+3+kind];if(length>trace_rows-needed){status[0]=12;break;}needed+=length;
    }
    if(!status[0])for(I event=0;event<c[1];++event)for(I kind=0;kind<2;++kind) {
      const I fiber=e[event*7],length=e[event*7+3+kind];
      auto k=(__gm__ float*)key,v=(__gm__ float*)value;
      for(I row=0;row<length;++row) {
        const I target=size[0]++;auto m=(__gm__ I*)meta+target*5;
        m[0]=f[fiber*4];m[1]=f[fiber*4+1];m[2]=f[fiber*4+2];m[3]=kind;m[4]=row;
        const I source=tide_device::event_key_row(e,event,row,capacity,owners,kind!=0);
        for(I d=0;d<width;++d){((__gm__ float*)trace)[target*2*width+d]=k[source*width+d];
          ((__gm__ float*)trace)[target*2*width+width+d]=v[source*width+d];}
      }
    }
  }else if(mode==1) {
    tide_device::FiberVector op;op.init();auto x=op.x();const I tiles=(width+255)/256;
    if(AscendC::GetBlockIdx()==0)for(I event=0;event<c[1];++event) {
      const I parameter=e[event*7+2],fiber=e[event*7];
      if(e[event*7+4]>maximum[0])maximum[0]=e[event*7+4];
      if(event+1<c[1]&&e[(event+1)*7+1]==e[event*7+1])continue;
      if(cfg[parameter*2]||on[fiber])n[e[event*7+1]]=cfg[parameter*2+1]&&on[fiber]?0:e[event*7+4];
    }
    for(I task=AscendC::GetBlockIdx();task<c[1]*tiles;task+=AscendC::GetBlockNum()) {
      const I event=task/tiles,start=(task%tiles)*256,parameter=e[event*7+2],owner=e[event*7+1],fiber=e[event*7];
      if(event+1<c[1]&&e[(event+1)*7+1]==owner)continue;
      if(!(cfg[parameter*2]||on[fiber])||(cfg[parameter*2+1]&&on[fiber]))continue;
      const I length=e[event*7+4];
      const uint32_t size=width-start<256?width-start:256;
      auto src_k=(__gm__ float*)key,src_v=(__gm__ float*)value;
      auto dst_k=(__gm__ float*)live_key,dst_v=(__gm__ float*)live_value;
      for(I row=0;row<length;++row) {
        const I src=tide_device::event_key_row(e,event,row,capacity,owners)*width+start,dst=(owner*capacity+row)*width+start;
        op.load(x,src_k,src,size);op.save(x,dst_k,dst,size);op.load(x,src_v,src,size);op.save(x,dst_v,dst,size);
      }
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
