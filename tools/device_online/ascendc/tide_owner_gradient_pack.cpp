#include "fiber_vector.h"
// Addresses are static descriptors built from retained tensors on this device.
// They are not a portable checkpoint format and never describe peer memory.
extern "C" __global__ __aicore__ void tide_owner_gradient_pack(GM_ADDR descriptors,GM_ADDR tiles,
    GM_ADDR values,GM_ADDR connected,GM_ADDR error,int64_t count,int64_t tasks) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)descriptors);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto d=(__gm__ I*)descriptors,off=(__gm__ I*)tiles;
  if(AscendC::GetBlockIdx()==0)for(I i=0;i<count;++i)
    ((__gm__ uint8_t*)connected)[i]=*(__gm__ uint8_t*)(uint64_t)d[i*4+1];
  tide_device::FiberVector op;op.init();auto x=op.x();
  for(I task=AscendC::GetBlockIdx();task<tasks;task+=AscendC::GetBlockNum()) {
    I lo=0,hi=count;while(lo+1<hi){const I mid=lo+(hi-lo)/2;if(off[mid]<=task)lo=mid;else hi=mid;}
    const auto row=d+lo*4;const I start=(task-off[lo])*256,remaining=row[3]-start;
    const uint32_t size=remaining<256?remaining:256;
    if(*(__gm__ uint8_t*)(uint64_t)row[1])op.load(x,(__gm__ float*)(uint64_t)row[0],start,size);
    else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
    op.save(x,(__gm__ float*)values,row[2]+start,size);
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
