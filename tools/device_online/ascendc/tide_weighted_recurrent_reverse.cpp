#include "weighted_recurrent_snapshot_math.h"
extern "C" __global__ __aicore__ void tide_weighted_recurrent_reverse(GM_ADDR selected,GM_ADDR meta,GM_ADDR decay,GM_ADDR fields,GM_ADDR bars,GM_ADDR roots,GM_ADDR bypass,GM_ADDR bypass_roots,GM_ADDR output,GM_ADDR connected,GM_ADDR error,int64_t rows,int64_t width,int64_t capacity,int64_t count){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_recurrent_device::reverse((__gm__ I*)selected,(__gm__ I*)meta,(__gm__ float*)decay,(__gm__ I*)fields,(__gm__ float*)bars,(__gm__ uint8_t*)roots,(__gm__ float*)bypass,(__gm__ uint8_t*)bypass_roots,(__gm__ float*)output,(__gm__ uint8_t*)connected,(__gm__ int32_t*)error,rows,width,capacity,count);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
