#include "weighted_allocation_math.h"
// One reusable numerical body visits actual nonempty banks on device. The
// capacity branch is shared by all banks, rather than recorded B times.
extern "C" __global__ __aicore__ void tide_weighted_bank_next(GM_ADDR buckets,GM_ADDR cursor,
    GM_ADDR branch,GM_ADDR error,int64_t banks,int64_t reset){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_weighted_device;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)cursor);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0){
    auto index=(__gm__ I*)cursor;auto selected=(__gm__ int32_t*)branch;auto err=(__gm__ int32_t*)error;
    selected[0]=0;
    if(reset)index[0]=-1;
    else if(!err[0]){
      if(index[0]<-1||index[0]>=banks){err[0]=2;}
      else{I b=index[0]+1;while(b<banks&&!((__gm__ int32_t*)buckets)[b])++b;
        index[0]=b;if(b<banks)selected[0]=((__gm__ int32_t*)buckets)[b];}
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
