#include "kernel_operator.h"
extern "C" __global__ __aicore__ void tide_control_merge(GM_ADDR source,GM_ADDR target,GM_ADDR error,int64_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<uint8_t> cache;cache.SetGlobalBuffer((__gm__ uint8_t*)source);
  AscendC::DataCacheCleanAndInvalid<uint8_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  for(int64_t i=0;i<size;++i)((__gm__ uint8_t*)target)[i]|=((__gm__ uint8_t*)source)[i];
  AscendC::DataCacheCleanAndInvalid<uint8_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
