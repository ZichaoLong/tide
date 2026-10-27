#include "weighted_convolution_math.h"
extern "C" __global__ __aicore__ void tide_weighted_conv_guard(GM_ADDR owners,GM_ADDR weights,GM_ADDR bias,GM_ADDR values,GM_ADDR roots,GM_ADDR clean,GM_ADDR flags,GM_ADDR error,int64_t rows,int64_t size,int64_t features,int64_t mode){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using I=int64_t;AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)owners);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_conv_device::guard((__gm__ I*)owners,(__gm__ float*)weights,(__gm__ float*)bias,(__gm__ float*)values,(__gm__ uint8_t*)roots,(__gm__ float*)clean,(__gm__ uint8_t*)flags,(__gm__ int32_t*)error,rows,size,features,mode);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
