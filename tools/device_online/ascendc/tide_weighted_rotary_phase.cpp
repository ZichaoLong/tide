#include "weighted_rotary_math.h"
extern "C" __global__ __aicore__ void tide_weighted_rotary_phase(GM_ADDR ages,GM_ADDR frequencies,GM_ADDR output,GM_ADDR error,int64_t count,int64_t pairs,float scale){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_rotary_device::phase((__gm__ I*)ages,(__gm__ float*)frequencies,(__gm__ float*)output,(__gm__ int32_t*)error,count,pairs,scale);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
