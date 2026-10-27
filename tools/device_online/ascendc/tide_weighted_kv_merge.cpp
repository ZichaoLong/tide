#include "weighted_kv_update_math.h"
extern "C" __global__ __aicore__ void tide_weighted_kv_merge(GM_ADDR mass,GM_ADDR mass_roots,GM_ADDR kv,GM_ADDR kv_roots,GM_ADDR summary,GM_ADDR summary_roots,GM_ADDR output,GM_ADDR connected,GM_ADDR error,int64_t rows,int64_t width){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using I=int64_t;AscendC::GlobalTensor<uint8_t> cache;cache.SetGlobalBuffer((__gm__ uint8_t*)output);
  AscendC::DataCacheCleanAndInvalid<uint8_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_kv_device::merge((__gm__ float*)mass,(__gm__ uint8_t*)mass_roots,(__gm__ float*)kv,(__gm__ uint8_t*)kv_roots,(__gm__ float*)summary,(__gm__ uint8_t*)summary_roots,(__gm__ float*)output,(__gm__ uint8_t*)connected,(__gm__ int32_t*)error,rows,width);
  AscendC::DataCacheCleanAndInvalid<uint8_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
