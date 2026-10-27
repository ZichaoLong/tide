#pragma once
// Architecture §2.2 shared roots: fixed alias metadata, actual dynamic roots.
#include "weighted_allocation_math.h"
#include "fiber_vector.h"
namespace tide_weighted_parameter_device {
using namespace tide_weighted_device;
__aicore__ inline void reduce(__gm__ I* owners,__gm__ I* refs,__gm__ I* tiles,
    __gm__ float* output,__gm__ uint8_t* connected,__gm__ int32_t* errors,
    __gm__ int32_t* error,I count,I tasks){
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer(owners);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(error[0])return;
  if(AscendC::GetBlockIdx()==0)for(I i=0;i<count;++i){bool present=false;
    for(I j=owners[i*4];j<owners[i*4+1];++j)present|=((__gm__ uint8_t*)(uint64_t)refs[2*j+1])[0]!=0;
    connected[i]=present;}
  tide_device::FiberVector vector;vector.init();auto sum=vector.x(),part=vector.y();
  for(I task=AscendC::GetBlockIdx();task<tasks;task+=AscendC::GetBlockNum()){
    I lo=0,hi=count;while(lo+1<hi){const I mid=lo+(hi-lo)/2;if(tiles[mid]<=task)lo=mid;else hi=mid;}
    const I i=lo,start=(task-tiles[i])*256,remaining=owners[i*4+3]-start;
    const uint32_t size=remaining<256?remaining:256;errors[task*16]=0;
    AscendC::Duplicate(sum,0.f,size);AscendC::PipeBarrier<PIPE_V>();
    for(I j=owners[i*4];j<owners[i*4+1];++j)if(((__gm__ uint8_t*)(uint64_t)refs[2*j+1])[0]){
      vector.load(part,(__gm__ float*)(uint64_t)refs[2*j],start,size);
      AscendC::Add(sum,sum,part,size);AscendC::PipeBarrier<PIPE_V>();}
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    for(uint32_t j=0;j<size;++j)if(!finite(sum.GetValue(j)))errors[task*16]=23;
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    vector.save(sum,output,owners[i*4+2]+start,size);
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
}
