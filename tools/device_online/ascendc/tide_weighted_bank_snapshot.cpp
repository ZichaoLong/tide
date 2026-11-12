#include "weighted_allocation_math.h"
// Copy each actual bank once. Equal values do not establish shared ownership.
extern "C" __global__ __aicore__ void tide_weighted_bank_snapshot(GM_ADDR connected,
    GM_ADDR source_matrix,GM_ADDR source_bias,GM_ADDR matrix,GM_ADDR bias,GM_ADDR error,
    int64_t banks,int64_t in,int64_t out,int64_t has_bias,int64_t error_code){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_weighted_device;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)connected);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  // Contiguous aligned chunks avoid different workers updating one cache line.
  for(I tile=AscendC::GetBlockIdx();tile<(banks*out*in+127)/128;tile+=AscendC::GetBlockNum())
    for(I j=tile*128;j<(tile+1)*128&&j<banks*out*in;++j){
      const float value=((__gm__ uint8_t*)connected)[j/(out*in)]?((__gm__ float*)source_matrix)[j]:0;
      ((__gm__ float*)matrix)[j]=value;if(!finite(value))((__gm__ int32_t*)error)[0]=error_code;
    }
  if(has_bias)for(I tile=AscendC::GetBlockIdx();tile<(banks*out+127)/128;tile+=AscendC::GetBlockNum())
    for(I j=tile*128;j<(tile+1)*128&&j<banks*out;++j){
      const float value=((__gm__ uint8_t*)connected)[j/out]?((__gm__ float*)source_bias)[j]:0;
      ((__gm__ float*)bias)[j]=value;if(!finite(value))((__gm__ int32_t*)error)[0]=error_code;
    }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
