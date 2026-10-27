#include "weighted_convolution_math.h"
extern "C" __global__ __aicore__ void tide_weighted_conv_plan(GM_ADDR old,GM_ADDR meta,GM_ADDR time,GM_ADDR current,GM_ADDR owners,GM_ADDR indices,GM_ADDR mapping,GM_ADDR taps,GM_ADDR error,int64_t rows,int64_t raw_width,int64_t capacity,int64_t offset,int64_t features,int64_t size,int64_t stride,int64_t banks){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using I=int64_t;AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)owners);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_conv_device::prepare((__gm__ float*)old,(__gm__ I*)meta,(__gm__ I*)time,(__gm__ float*)current,(__gm__ I*)owners,(__gm__ I*)indices,(__gm__ I*)mapping,(__gm__ float*)taps,(__gm__ int32_t*)error,rows,raw_width,capacity,offset,features,size,stride,banks);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
