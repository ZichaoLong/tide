#include "weighted_row_graph_math.h"
extern "C" __global__ __aicore__ void tide_weighted_row_bind(GM_ADDR owners,GM_ADDR input,GM_ADDR output,GM_ADDR error,int64_t rows,int64_t width,int64_t banks,int64_t positive){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_row_device::bind((__gm__ I*)owners,(__gm__ float*)input,(__gm__ float*)output,(__gm__ int32_t*)error,rows,width,banks,positive);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
