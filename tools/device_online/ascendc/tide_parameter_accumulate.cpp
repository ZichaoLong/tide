#include "fiber_vector.h"
extern "C" __global__ __aicore__ void tide_parameter_accumulate(GM_ADDR owners,GM_ADDR tiles,
    GM_ADDR left,GM_ADDR left_connected,GM_ADDR right,GM_ADDR right_connected,GM_ADDR output,GM_ADDR connected,GM_ADDR error,int64_t count,int64_t tasks) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)owners);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto table=(__gm__ I*)owners,offsets=(__gm__ I*)tiles;auto ac=(__gm__ uint8_t*)left_connected,bc=(__gm__ uint8_t*)right_connected;
  if(AscendC::GetBlockIdx()==0)for(I i=0;i<count;++i) {
    const bool on=ac[i]||bc[i];((__gm__ uint8_t*)connected)[i]=on;
    if(on&&table[i*2]<0)((__gm__ int32_t*)error)[0]=2;
  }
  tide_device::FiberVector vector;vector.init();auto x=vector.x(),y=vector.y();
  for(I task=AscendC::GetBlockIdx();task<tasks;task+=AscendC::GetBlockNum()) {
    I lo=0,hi=count;while(lo+1<hi){I mid=lo+(hi-lo)/2;if(offsets[mid]<=task)lo=mid;else hi=mid;}
    const I i=lo,start=(task-offsets[i])*256,offset=table[i*2]+start,remaining=table[i*2+1]-start;
    const uint32_t size=remaining<256?remaining:256;AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();
    if(ac[i])vector.load(x,(__gm__ float*)left,offset,size);
    if(bc[i]){vector.load(y,(__gm__ float*)right,offset,size);AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();}
    vector.save(x,(__gm__ float*)output,offset,size);
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
