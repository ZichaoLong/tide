#include "weighted_convolution_math.h"
extern "C" __global__ __aicore__ void tide_weighted_conv_reverse(GM_ADDR owners,GM_ADDR meta,GM_ADDR mapping,GM_ADDR taps_bar,GM_ADDR weights_bar,GM_ADDR bias_bar,GM_ADDR roots,GM_ADDR old_bar,GM_ADDR current_bar,GM_ADDR cache_roots,GM_ADDR weight_out,GM_ADDR bias_out,GM_ADDR owner_roots,GM_ADDR error,int64_t rows,int64_t raw_width,int64_t capacity,int64_t offset,int64_t features,int64_t size,int64_t banks){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using I=int64_t;AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)owners);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_conv_device::reverse((__gm__ I*)owners,(__gm__ I*)meta,(__gm__ I*)mapping,(__gm__ float*)taps_bar,(__gm__ float*)weights_bar,(__gm__ float*)bias_bar,(__gm__ uint8_t*)roots,(__gm__ float*)old_bar,(__gm__ float*)current_bar,(__gm__ uint8_t*)cache_roots,(__gm__ float*)weight_out,(__gm__ float*)bias_out,(__gm__ uint8_t*)owner_roots,(__gm__ int32_t*)error,rows,raw_width,capacity,offset,features,size,banks);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
