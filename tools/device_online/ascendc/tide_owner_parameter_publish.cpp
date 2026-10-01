#include "fiber_vector.h"
// Destinations are retained local bank addresses, never peer pointers. Tiles
// preserve disjoint rows, including strided Q/K/V and short scalar tails.
extern "C" __global__ __aicore__ void tide_owner_parameter_publish(GM_ADDR descriptors,GM_ADDR tiles,
    GM_ADDR values,GM_ADDR error,int64_t count,int64_t tasks) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)descriptors);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto d=(__gm__ I*)descriptors,off=(__gm__ I*)tiles;tide_device::FiberVector op;op.init();auto x=op.x();
  for(I task=AscendC::GetBlockIdx();task<tasks;task+=AscendC::GetBlockNum()) {
    I lo=0,hi=count;while(lo+1<hi){I mid=lo+(hi-lo)/2;if(off[mid]<=task)lo=mid;else hi=mid;}
    const auto row=d+lo*7;const I per=(row[3]+255)/256,r=(task-off[lo])/per,start=(task-off[lo])%per*256;
    const uint32_t size=row[3]-start<256?row[3]-start:256;
    op.load(x,(__gm__ float*)values,row[0]+r*row[3]+start,size);
    if(row[5])op.save(x,(__gm__ half*)(uint64_t)row[1],r*row[4]+start,size);
    else {
      if(row[6]) {
        auto rounded=op.y().ReinterpretCast<half>();AscendC::Cast(rounded,x,AscendC::RoundMode::CAST_RINT,size);AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(x,rounded,AscendC::RoundMode::CAST_NONE,size);AscendC::PipeBarrier<PIPE_V>();
      }
      op.save(x,(__gm__ float*)(uint64_t)row[1],r*row[4]+start,size);
    }
  }
}
