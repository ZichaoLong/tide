#include "weighted_allocation_math.h"
// Gather a bounded actual bank, or scatter its numerical result. Padding is
// structural absence; unselected rows/poisoned cotangents are never read.
extern "C" __global__ __aicore__ void tide_weighted_bank_rows(GM_ADDR permutation,GM_ADDR offsets,
    GM_ADDR bank_id,GM_ADDR source,GM_ADDR destination,GM_ADDR error,int64_t capacity,
    int64_t width,int64_t scatter,int64_t error_code){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_weighted_device;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)offsets);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  const I bank=((__gm__ I*)bank_id)[0];
  const I begin=((__gm__ I*)offsets)[bank],count=((__gm__ I*)offsets)[bank+1]-begin;
  if(count<0||count>capacity){((__gm__ int32_t*)error)[0]=2;return;}
  for(I j=AscendC::GetBlockIdx();j<capacity*width;j+=AscendC::GetBlockNum()){
    const I r=j/width,col=j%width;
    if(scatter){if(r<count){const float value=((__gm__ float*)source)[j];
      if(!finite(value))((__gm__ int32_t*)error)[0]=error_code;
      ((__gm__ float*)destination)[((__gm__ I*)permutation)[begin+r]*width+col]=value;}}
    else{const float value=r<count?((__gm__ float*)source)[((__gm__ I*)permutation)[begin+r]*width+col]:0;
      if(!finite(value))((__gm__ int32_t*)error)[0]=error_code;((__gm__ float*)destination)[j]=value;}
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
