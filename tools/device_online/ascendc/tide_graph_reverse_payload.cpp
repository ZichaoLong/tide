#include "fiber_vector.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_graph_reverse_payload(GM_ADDR values,GM_ADDR fiber_values,GM_ADDR messages,
    GM_ADDR scales,GM_ADDR producer_head,GM_ADDR producer_next,GM_ADDR consumer_head,GM_ADDR consumer_next,
    GM_ADDR pending_root,GM_ADDR output_root,GM_ADDR final_root,GM_ADDR gradients,GM_ADDR connected,GM_ADDR carry,
    GM_ADDR carry_connected,GM_ADDR stage_values,GM_ADDR full_gradient,GM_ADDR range,GM_ADDR count,GM_ADDR cotangents,
    GM_ADDR cot_connected,GM_ADDR full_h,GM_ADDR full_c,GM_ADDR state_h,GM_ADDR state_connected,GM_ADDR aggregate_partials,
    GM_ADDR error,int64_t width,int64_t fibers,int64_t pending,int64_t outputs,int64_t nodes,int64_t samples,int64_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)messages);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto msg=(__gm__ I*)messages,ph=(__gm__ I*)producer_head,pn=(__gm__ I*)producer_next,ch=(__gm__ I*)consumer_head,cn=(__gm__ I*)consumer_next;
  auto on=(__gm__ uint8_t*)connected,cc=(__gm__ uint8_t*)cot_connected;
  auto g=(__gm__ float*)gradients,weights=(__gm__ float*)scales;
  const I tiles=(width+255)/256,rows=mode==0?fibers+pending+outputs+samples*nodes:((__gm__ I*)count)[0];
  const I first=((__gm__ I*)range)[0];
  tide_device::FiberVector vector;vector.init();auto x=vector.x(),y=vector.y();
  for(I task=AscendC::GetBlockIdx();task<rows*tiles;task+=AscendC::GetBlockNum()) {
    const I row=task/tiles,start=(task%tiles)*256;const uint32_t size=width-start<256?width-start:256;
    if(mode==0) {
      if(row<fibers)continue;
      if(row<fibers+pending+outputs) {
        if(!on[row])continue;const bool is_pending=row<fibers+pending;const I local=row-fibers-(is_pending?0:pending);
        vector.load(x,(__gm__ float*)(is_pending?pending_root:output_root),local*width+start,size);vector.save(x,g,row*width+start,size);
      } else {const I key=row-fibers-pending-outputs;if(!((__gm__ uint8_t*)carry_connected)[key])continue;
        vector.load(x,(__gm__ float*)final_root,key*width+start,size);vector.save(x,(__gm__ float*)carry,key*width+start,size);}
    } else if(mode==1) {
      const I event=first+row;
      for(I field=0;field<5;++field){vector.load(x,(__gm__ float*)values,event*(5*width+2)+field*width+start,size);
        vector.save(x,(__gm__ float*)stage_values,row*(5*width+2)+field*width+start,size);}
      AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();
      for(I m=ph[event];m>=0;m=pn[m])if(on[m]) {
        const float scale=weights[msg[m*4+3]];
        vector.load(y,g,m*width+start,size);AscendC::Muls(y,y,scale,size);AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();
      }
      vector.save(x,(__gm__ float*)full_gradient,row*width+start,size);
    } else if(mode==2) {
      for(I field=0;field<5;++field) {
        if(cc[row*5+field])vector.load(x,(__gm__ float*)(field==0?full_h:full_c),row*width+start,size);
        else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
        vector.save(x,(__gm__ float*)cotangents,(row*5+field)*width+start,size);
      }
    } else if(mode==3) {
      if(!((__gm__ uint8_t*)state_connected)[row])continue;
      vector.load(x,(__gm__ float*)state_h,row*width+start,size);
      for(I m=ch[first+row];m>=0;m=cn[m]) {
        const float scale=weights[msg[m*4+2]];
        AscendC::Muls(y,x,scale,size);AscendC::PipeBarrier<PIPE_V>();vector.save(y,g,m*width+start,size);
        vector.load(y,(__gm__ float*)fiber_values,m*width+start,size);AscendC::Mul(y,x,y,size);AscendC::PipeBarrier<PIPE_V>();
        vector.save(y,(__gm__ float*)aggregate_partials,m*width+start,size);
      }
    }
  }
}
