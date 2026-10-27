#include "weighted_raw_update_math.h"
extern "C" __global__ __aicore__ void tide_weighted_raw_apply(GM_ADDR fields,GM_ADDR old,GM_ADDR proposal,GM_ADDR mapping,GM_ADDR receive,GM_ADDR written,GM_ADDR output,GM_ADDR error,int64_t rows,int64_t width,int64_t capacity,int64_t count,int64_t convolution){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using I=int64_t;AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)fields);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_raw_device::apply((__gm__ I*)fields,(__gm__ float*)old,(__gm__ float*)proposal,(__gm__ I*)mapping,(__gm__ float*)receive,(__gm__ uint8_t*)written,(__gm__ float*)output,(__gm__ int32_t*)error,rows,width,capacity,count,convolution);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
