#include "fiber_vector.h"
extern "C" __global__ __aicore__ void tide_event_publish(GM_ADDR plan,GM_ADDR tiles,GM_ADDR source,
    GM_ADDR qkv,GM_ADDR projection,GM_ADDR error,int64_t count,int64_t tasks) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)plan);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto p=(__gm__ I*)plan,off=(__gm__ I*)tiles;tide_device::FiberVector op;op.init();auto x=op.x();
  for(I task=AscendC::GetBlockIdx();task<tasks;task+=AscendC::GetBlockNum()) {
    I lo=0,hi=count;while(lo+1<hi){const I mid=lo+(hi-lo)/2;if(off[mid]<=task)lo=mid;else hi=mid;}
    auto row=p+lo*6;const I tile=task-off[lo],cols=row[4],per=(cols+255)/256,r=tile/per,start=(tile%per)*256;
    const uint32_t size=cols-start<256?cols-start:256;
    op.load(x,(__gm__ float*)source,row[0]+r*cols+start,size);
    op.save(x,(__gm__ float*)(row[1]?projection:qkv),row[2]+r*row[5]+start,size);
  }
}
