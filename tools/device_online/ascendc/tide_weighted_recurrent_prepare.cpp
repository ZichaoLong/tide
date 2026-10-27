#include "weighted_recurrent_snapshot_math.h"
extern "C" __global__ __aicore__ void tide_weighted_recurrent_prepare(GM_ADDR raw,GM_ADDR meta,GM_ADDR time,GM_ADDR mass,GM_ADDR owners,GM_ADDR present,GM_ADDR rates,GM_ADDR clean,GM_ADDR decay,GM_ADDR selected,GM_ADDR error,int64_t rows,int64_t width,int64_t capacity,int64_t d,int64_t matrices,int64_t matrix_width,int64_t ds,int64_t cf,int64_t banks,int64_t positive,int64_t horizon){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_recurrent_device::prepare((__gm__ float*)raw,(__gm__ I*)meta,(__gm__ I*)time,(__gm__ float*)mass,(__gm__ I*)owners,(__gm__ uint8_t*)present,(__gm__ float*)rates,(__gm__ float*)clean,(__gm__ float*)decay,(__gm__ I*)selected,(__gm__ int32_t*)error,rows,width,capacity,d,matrices,matrix_width,ds,cf,banks,positive,horizon);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
