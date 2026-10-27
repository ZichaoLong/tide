#include "weighted_row_graph_math.h"
extern "C" __global__ __aicore__ void tide_weighted_row_accumulate(GM_ADDR owners,GM_ADDR input,GM_ADDR roots,GM_ADDR output,GM_ADDR connected,GM_ADDR error,int64_t rows,int64_t width){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_row_device::accumulate((__gm__ I*)owners,(__gm__ float*)input,(__gm__ uint8_t*)roots,(__gm__ float*)output,(__gm__ uint8_t*)connected,(__gm__ int32_t*)error,rows,width);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
