#include "fiber_vector.h"
// Sum complete logical owners in declared reverse-window/alias order. Each
// feature tile has one writer; None remains separate from connected zero.
extern "C" __global__ __aicore__ void tide_owner_gradient_reduce(GM_ADDR owners,GM_ADDR references,GM_ADDR tiles,
    GM_ADDR partials,GM_ADDR partial_connected,GM_ADDR output,GM_ADDR output_connected,GM_ADDR error,
    int64_t count,int64_t tasks) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)owners);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto owner=(__gm__ I*)owners,refs=(__gm__ I*)references,off=(__gm__ I*)tiles;
  auto on=(__gm__ uint8_t*)partial_connected;
  if(AscendC::GetBlockIdx()==0)for(I i=0;i<count;++i) {
    bool live=false;for(I j=owner[i*4];j<owner[i*4+1];++j)live|=on[refs[j*2+1]];
    ((__gm__ uint8_t*)output_connected)[i]=live;
  }
  tide_device::FiberVector op;op.init();auto total=op.x(),part=op.y();
  for(I task=AscendC::GetBlockIdx();task<tasks;task+=AscendC::GetBlockNum()) {
    I lo=0,hi=count;while(lo+1<hi){const I mid=lo+(hi-lo)/2;if(off[mid]<=task)lo=mid;else hi=mid;}
    const auto row=owner+lo*4;const I start=(task-off[lo])*256,remaining=row[3]-start;
    const uint32_t size=remaining<256?remaining:256;
    AscendC::Duplicate(total,0.f,size);AscendC::PipeBarrier<PIPE_V>();
    for(I j=row[0];j<row[1];++j)if(on[refs[j*2+1]]) {
      op.load(part,(__gm__ float*)partials,refs[j*2]+start,size);AscendC::Add(total,total,part,size);AscendC::PipeBarrier<PIPE_V>();
    }
    op.save(total,(__gm__ float*)output,row[2]+start,size);
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
