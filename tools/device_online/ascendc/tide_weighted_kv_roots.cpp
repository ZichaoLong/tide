#include "weighted_kv_update_math.h"
extern "C" __global__ __aicore__ void tide_weighted_kv_roots(GM_ADDR raw,GM_ADDR kv,GM_ADDR summary,GM_ADDR error,int64_t rows){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using I=int64_t;AscendC::GlobalTensor<uint8_t> cache;cache.SetGlobalBuffer((__gm__ uint8_t*)raw);
  AscendC::DataCacheCleanAndInvalid<uint8_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_kv_device::roots((__gm__ uint8_t*)raw,(__gm__ uint8_t*)kv,(__gm__ uint8_t*)summary,(__gm__ int32_t*)error,rows);
  AscendC::DataCacheCleanAndInvalid<uint8_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
