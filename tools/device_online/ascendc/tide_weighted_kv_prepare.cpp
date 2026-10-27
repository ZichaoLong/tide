#include "weighted_kv_update_math.h"
extern "C" __global__ __aicore__ void tide_weighted_kv_prepare(GM_ADDR h,GM_ADDR owners,GM_ADDR present,GM_ADDR u,GM_ADDR mass,GM_ADDR selected,GM_ADDR indices,GM_ADDR row_ids,GM_ADDR positive,GM_ADDR error,int64_t rows,int64_t width,int64_t banks){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using I=int64_t;AscendC::GlobalTensor<uint8_t> cache;cache.SetGlobalBuffer((__gm__ uint8_t*)owners);
  AscendC::DataCacheCleanAndInvalid<uint8_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_kv_device::prepare((__gm__ float*)h,(__gm__ I*)owners,(__gm__ uint8_t*)present,(__gm__ float*)u,(__gm__ float*)mass,(__gm__ I*)selected,(__gm__ I*)indices,(__gm__ I*)row_ids,(__gm__ uint8_t*)positive,(__gm__ int32_t*)error,rows,width,banks);
  AscendC::DataCacheCleanAndInvalid<uint8_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
