#include "fiber_vector.h"
extern "C" __global__ __aicore__ void tide_parameter_publish(GM_ADDR plan,GM_ADDR tiles,GM_ADDR values,GM_ADDR weights,GM_ADDR biases,
    GM_ADDR decay,GM_ADDR retention,GM_ADDR read,GM_ADDR sources,GM_ADDR emission,GM_ADDR error,int64_t count,int64_t tasks) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)plan);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto table=(__gm__ I*)plan,offset=(__gm__ I*)tiles;tide_device::FiberVector vector;vector.init();auto x=vector.x();
  for(I task=AscendC::GetBlockIdx();task<tasks;task+=AscendC::GetBlockNum()) {
    I lo=0,hi=count;while(lo+1<hi){I mid=lo+(hi-lo)/2;if(offset[mid]<=task)lo=mid;else hi=mid;}
    const I i=lo,start=(task-offset[i])*256,remaining=table[i*4+3]-start,bank=table[i*4+1];
    const uint32_t size=remaining<256?remaining:256;
    auto destination=(__gm__ float*)(bank==0?weights:bank==1?biases:bank==2?decay:bank==3?retention:bank==4?read:bank==5?sources:emission);
    vector.load(x,(__gm__ float*)values,table[i*4]+start,size);vector.save(x,destination,table[i*4+2]+start,size);
  }
}
