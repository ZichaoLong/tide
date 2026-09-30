#include "fiber_vector.h"
#include "../optimizer_layout.h"
extern "C" __global__ __aicore__ void tide_optimizer_commit(GM_ADDR table,GM_ADDR tiles,GM_ADDR options,GM_ADDR flags,
    GM_ADDR connected,GM_ADDR values,GM_ADDR first,GM_ADDR second,GM_ADDR maximum,GM_ADDR steps,GM_ADDR corrections,
    GM_ADDR next_values,GM_ADDR next_first,GM_ADDR next_second,GM_ADDR next_maximum,GM_ADDR next_steps,GM_ADDR next_corrections,
    GM_ADDR error,int64_t count,int64_t tasks,int64_t kind) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_device;using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)table);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto t=(__gm__ I*)table,offsets=(__gm__ I*)tiles,fl=(__gm__ I*)flags;
  if(AscendC::GetBlockIdx()==0)for(I i=0;i<count;++i)
    if(t[i*OWNER_FIELDS+OFFSET]>=0&&t[i*OWNER_FIELDS+GROUP]>=0&&((__gm__ uint8_t*)connected)[i]) {
      ((__gm__ I*)steps)[i]=((__gm__ I*)next_steps)[i];
      if(kind){((__gm__ float*)corrections)[i*2]=((__gm__ float*)next_corrections)[i*2];
        ((__gm__ float*)corrections)[i*2+1]=((__gm__ float*)next_corrections)[i*2+1];}
    }
  FiberVector vector;vector.init();auto x=vector.x();
  for(I task=AscendC::GetBlockIdx();task<tasks;task+=AscendC::GetBlockNum()) {
    I lo=0,hi=count;while(lo+1<hi){I mid=lo+(hi-lo)/2;if(offsets[mid]<=task)lo=mid;else hi=mid;}
    const I i=lo;if(!((__gm__ uint8_t*)connected)[i])continue;
    const I start=(task-offsets[i])*256,offset=t[i*OWNER_FIELDS+OFFSET]+start,remaining=t[i*OWNER_FIELDS+SIZE]-start;
    const uint32_t size=remaining<256?remaining:256;const I group=t[i*OWNER_FIELDS+GROUP];
    vector.load(x,(__gm__ float*)next_values,offset,size);vector.save(x,(__gm__ float*)values,offset,size);
    if(kind||((__gm__ float*)options)[group*OPTION_COUNT+MOM]!=0) {
      vector.load(x,(__gm__ float*)next_first,offset,size);vector.save(x,(__gm__ float*)first,offset,size);
    }
    if(kind) {
      vector.load(x,(__gm__ float*)next_second,offset,size);vector.save(x,(__gm__ float*)second,offset,size);
      if(fl[group*FLAG_COUNT+AMSGRAD]){vector.load(x,(__gm__ float*)next_maximum,offset,size);vector.save(x,(__gm__ float*)maximum,offset,size);}
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
