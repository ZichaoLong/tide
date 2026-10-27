#include "weighted_kv_window_math.h"
extern "C" __global__ __aicore__ void tide_weighted_window_prepare(GM_ADDR raw,GM_ADDR meta,GM_ADDR times,GM_ADDR present,GM_ADDR keys,GM_ADDR values,GM_ADDR summaries,GM_ADDR weights,GM_ADDR decay,GM_ADDR ages,GM_ADDR mapping,GM_ADDR lengths,GM_ADDR selected,GM_ADDR error,int64_t rows,int64_t capacity,int64_t d,int64_t ds,int64_t horizon,float beta){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_window_device::prepare((__gm__ float*)raw,(__gm__ I*)meta,(__gm__ I*)times,(__gm__ uint8_t*)present,(__gm__ float*)keys,(__gm__ float*)values,(__gm__ float*)summaries,(__gm__ float*)weights,(__gm__ float*)decay,(__gm__ I*)ages,(__gm__ I*)mapping,(__gm__ I*)lengths,(__gm__ uint8_t*)selected,(__gm__ int32_t*)error,rows,capacity,d,ds,horizon,beta);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
